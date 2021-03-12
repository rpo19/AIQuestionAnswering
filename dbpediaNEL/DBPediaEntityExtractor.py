import spacy_dbpedia_spotlight
import spacy

class DBPediaEntityExtractor():
    __init__(self):
        # load model and keep only ner
        nlp = spacy.load('en_core_web_lg', 
                        disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        # add dbpedia-spotlight stage
        nlp.add_pipe('dbpedia_spotlight', config={'overwrite_ents': True})
    
    def extract(text):
        # execute NER and NEL
        doc = nlp(text)
        nel_ents = doc.ents
        
        # filter entities
        filtered_ents = []
        for nel_ent in nel_ents:
            # if there are NER ents
            try:
                ner_ents = doc.spans['ents_original']
                for ner_ent in ner_ents:
                    # keep only entities extracted with both spacy's NER and dbpedia-spotlight
                    if ner_ent.text == nel_ent.text:
                        filtered_ents.append((nel_ent.kb_id_, nel_ent.text))
            except:
                # no NER ents, keep all the dbpedia-spotlight ones
                filtered_ents.append((nel_ent.kb_id_, nel_ent.text))
        
        return filtered_ents