import spacy_dbpedia_spotlight
import spacy
import requests
import flair
from pathlib import Path
flair.cache_root = Path('./data/flair')
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from scipy import spatial
from Levenshtein import distance as levenshtein_distance


class DBPediaEntityExtractor():
    def __init__(self, mode='spotlight'):
        self.mode = mode

        if mode == 'spotlight':
            # load model and keep only ner
            print('Loading \'en_core_web_lg\' model...')
            self.nlp = spacy.load('en_core_web_lg')
            # add dbpedia-spotlight stage
            # overwrite_ents = False means we have to use doc.spans['dbpedia_ents']
            self.nlp.add_pipe('dbpedia_spotlight', config={'overwrite_ents': False})
        elif mode == 'custom':
            print('Loading flair NER models...')
            # load NER model
            self.tagger = SequenceTagger.load('ner-fast')
            # load sentence embedding model
            self.embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

    """
    Get text sentence level embedding.
    """
    def __get_text_embedding(self, text):
        sentence = Sentence(text)
        self.embedding.embed(sentence)
        return sentence.embedding.tolist()

    """
    Extract entities from text.
    """
    def __extract_entities(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence)

        entities = sentence.to_dict(tag_type='ner')['entities']
        entities = [entity['text'] for entity in entities]
        return entities

    """
    Create tuples from a list with overlapping elements.
    (e.g: [1,2,3] -> [(1,2), (2,3)])
    """
    def __split_overlap(self, seq, size, overlap):
        return [x for x in zip(*[seq[i::size-overlap] for i in range(size)])]

    """
    Extend entity phrase with its neighbour.
    """
    def __extend_entity(self, text, phr, max_len):
        tmp_text = text.replace(phr, 'ENTITY')
        # get question tokens
        text_tokens = tmp_text.split()
        # get position of current entity
        index = text_tokens.index('ENTITY')

        extended_entities = []

        for size in range(1, max_len+1):
            for group in self.__split_overlap(text_tokens, size, size-1):
                if 'ENTITY' in group:
                    extended_entities.append(' '.join(group).replace('ENTITY', phr))
        return extended_entities

    """
    Get query lookup results.
    """
    def __lookup(self, phr, max_res = 10):
        res = requests.get(f'https://lookup.dbpedia.org/api/search?query={phr}&maxResults=10&format=JSON_RAW')
        docs = eval(res.text)['docs']
        return docs

    """
    Compute relevance between entity phrase and candidate entity.
    score = alfa1 * importance + alfa2 * lev_distance + alfa3 * cos_sim
    """
    def __compute_relevance(self, phr, candidate_entity, text_embedding, rank, alfa1=1, alfa2=1, alfa3=1):
        # TODO: compute importance
        # can we use the relevance or simply the rank of results from lookup?
        importance = 1 / rank

        # compute lev distance
        lev_distance = 1 / (levenshtein_distance(phr, candidate_entity['label'][0]) + 1)

        # compute relevance with doc embedding
        if 'comment' in candidate_entity:
            doc__entity_embedding = self.__get_text_embedding(candidate_entity['comment'][0])
            cos_sim = 1 - spatial.distance.cosine(text_embedding, doc__entity_embedding)
        else:
            cos_sim = 0

        score = alfa1 * importance + alfa2 * lev_distance + alfa3 * cos_sim
        return score

    """
    Extract and link entities from a text as described from the paper.
    """
    def __extract_custom(self, text, max_len = 3):
        text = text.replace('?', ' ?')

        entities_URIs = []
        entities_texts = []
        entities_scores = []

        # get text embedding
        text_embedding = self.__get_text_embedding(text)

        # extract entities from question
        entity_phrases = self.__extract_entities(text)

        # iterate for each extracted entity
        for i, phr in enumerate(entity_phrases):
            candidate_entity_phrase = {'phr': phr, 'candidate_entity': None, 'score': 0}

            # extend extracted entities
            PX = self.__extend_entity(text, phr, max_len)
            EC = []
            ranks = []
            # look for candidate entities
            for phr_ext in PX:
                docs = self.__lookup(phr_ext)
                # if there is at least a match add to candidate entities
                if len(docs) > 0:
                    EC.extend(docs)
                    ranks.extend(list(range(1, len(docs) + 1)))
            # compute relevances and keep highest relevance candidate entity
            for j, candidate_entity in enumerate(EC):
                tmp_score = self.__compute_relevance(phr, candidate_entity, text_embedding, ranks[j])
                if tmp_score > candidate_entity_phrase['score']:
                    candidate_entity_phrase['candidate_entity'] = candidate_entity
                    candidate_entity_phrase['score'] = tmp_score

            entities_URIs.append('<'+candidate_entity_phrase['candidate_entity']['resource'][0]+'>')
            entities_texts.append(candidate_entity_phrase['phr'])
            entities_scores.append(candidate_entity_phrase['score'])

        return entities_URIs, entities_texts, entities_scores

    """
    Extract and link entities from a text with DBPedia Spotlight.
    """
    def __spotlight_extract(self, text):
        # possessive forms may induce problems
        text = text.replace('\'s ', ' ')
        # execute NER and NEL
        disable = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']
        doc = self.nlp(text, disable=disable)
        nel_ents = doc.spans['dbpedia_ents']
        # filter entities
        filtered_ents_uri = []
        filtered_ents_text = []
        for nel_ent in nel_ents:
            # if there are NER ents
            try:
                ner_ents = doc.spans['ents_original']
                for ner_ent in ner_ents:
                    # keep only entities extracted with both spacy's NER and dbpedia-spotlight
                    if ner_ent.text == nel_ent.text:
                        filtered_ents_uri.append('<'+nel_ent.kb_id_+'>')
                        filtered_ents_text.append(nel_ent.text)
            except:
                # no NER ents, keep all the dbpedia-spotlight ones
                filtered_ents_uri.append('<'+nel_ent.kb_id_+'>')
                filtered_ents_text.append(nel_ent.text)

        return filtered_ents_uri, filtered_ents_text

    def extract(self, text):
        if self.mode == 'spotlight':
            return self.__spotlight_extract_v2(text)
        elif self.mode == 'custom':
            return self.__extract_custom(text)

    def __spotlight_extract_v2(self, text):
        # possessive forms may induce problems
        text = text.replace('\'s ', ' ')
        # execute NER and NEL
        disable = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']
        doc = self.nlp(text, disable=disable)
        nel_ents = doc.spans['dbpedia_ents']

        # filter entities
        filtered_ents_uri = []
        filtered_ents_text = []
        for nel_ent in nel_ents:
            if nel_ent.text[0].isupper():
                filtered_ents_uri.append('<'+nel_ent.kb_id_+'>')
                filtered_ents_text.append(nel_ent.text)
            else:
                try:
                    ner_ents = doc.spans['ents_original'] if 'ents_original' in doc.spans else doc.ents
                    for ner_ent in ner_ents:
                        # keep only entities extracted with both spacy's NER and dbpedia-spotlight
                        if ner_ent.text == nel_ent.text:
                            filtered_ents_uri.append('<'+nel_ent.kb_id_+'>')
                            filtered_ents_text.append(nel_ent.text)
                except Exception as e:
                    print('Got exception', e)
                    continue
        # if filter was too strict it's ok to keep all entities
        if len(filtered_ents_uri) == 0:
            for nel_ent in nel_ents:
                filtered_ents_uri.append('<'+nel_ent.kb_id_+'>')
                filtered_ents_text.append(nel_ent.text)

        return filtered_ents_uri, filtered_ents_text

    """
    Extract only the last entity
    """
    def extractLast(self, text):
        ents = self.__spotlight_extract_v2(text)
        if len(ents[0]) > 0:
            return ents[0][-1], ents[1][-1]
        else:
            return ents

    """
    Extract only the main entity
    """
    def extractMain(self, text):
        # extract entities
        ents = self.__spotlight_extract_v2(text)
        # extract tagged noun chunks
        disable = ['dbpedia_spotlight']
        doc = self.nlp(text, disable=disable)
        # search for main entity
        previous_dep_ = None
        for chunk in doc.noun_chunks:
            # main entity appears as subject or propositional/direct object next to a subject
            if chunk.root.dep_ == 'nsubj' or (chunk.root.dep_ in ['pobj', 'dobj'] and previous_dep_ == 'nsubj'):
                for i, ent in enumerate(ents[1]):
                    if ent in chunk.text or chunk.text in ent:
                        return ents[0][i], ent
            previous_dep_ = chunk.root.dep_
        # return last entity in case of no main entity found
        if len(ents[0]) > 0:
            return ents[0][-1], ents[1][-1]
        else:
            return ents