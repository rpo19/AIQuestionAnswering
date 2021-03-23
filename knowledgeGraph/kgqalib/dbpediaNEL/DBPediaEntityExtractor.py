import spacy_dbpedia_spotlight
import spacy

class DBPediaEntityExtractor():
    def __init__(self):
        # load model and keep only ner
        print('Loading \'en_core_web_lg\' model...')
        self.nlp = spacy.load('en_core_web_lg', 
                        disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        # add dbpedia-spotlight stage
        self.nlp.add_pipe('dbpedia_spotlight', config={'overwrite_ents': True})
    
    def extract(self, text):
        # execute NER and NEL
        doc = self.nlp(text)
        nel_ents = doc.ents
        
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
                        ent = {
                            'id': nel_ent.kb_id_,
                            'text': nel_ent.text
                        }
                        filtered_ents.append()
            except:
                # no NER ents, keep all the dbpedia-spotlight ones
                filtered_ents_uri.append('<'+nel_ent.kb_id_+'>')
                filtered_ents_text.append(nel_ent.text)
        
        return filtered_ents_uri, filtered_ents_text