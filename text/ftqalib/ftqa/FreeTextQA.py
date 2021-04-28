from transformers import pipeline
import wikipediaapi


class FreeTextQA():
    def __init__(self):
        self.__USELESS_SECTIONS = ['External links', 'See also', 'Notes', 'References', 'Bibliography', 'Further reading']
        # load ftqa model
        # the model is pretrained on SQuAD by default, but you can either specify it
        # model_name = "deepset/roberta-base-squad2"
        print('Loading question-answering model...')
        self.__qa = pipeline("question-answering")
        # load wikipedia api
        self.__wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI
            )

    def answerFromWiki(self, question, entity, debug=False): 
        # check entity
        if len(entity) == 0:
                return {'entity':'Entity not provided!', 'answer': None}
        # extract resource name from entity
        entity = entity.split('/')[-1].replace('>','')
        # get wikipedia page
        p = self.__wiki.page(entity)
        # get flattened sections of the page
        sections = self.__flattenSections(p)
        # iterate on sections and find answer in each span
        max_score = 0
        best_answer = None

        for section in sections:
                # TODO: how to identify a smaller span? (check if it's necessary...)
                # if subject in section.text: # look only in relevant sections???
                context = section.text
                if context != '':
                        answer = self.answerFromSpan(question=question, context=context)
                        if answer['score'] > max_score:
                                max_score = answer['score']
                                best_answer = answer
                                best_answer['entity'] = entity
                                best_answer['section'] = section.title
                        if debug:
                                tokens = self.__qa.tokenizer(context)['input_ids']
                                print(section.title, f'(length: {len(tokens)}')
                                print(f"Answer: '{answer['answer']}' with score {answer['score']}", '\n', 30*'_')
        return best_answer

    def answerFromSpan(self, question, context):
        return self.__qa(question, context)


    # recursively flatten wiki page sections
    def __flattenSections(self, section):
            if section.title not in self.__USELESS_SECTIONS:
                    sections = []
                    if len(section.sections) == 0:
                            return [section]
                    else:
                            # it has a section introduction that could contain the answer
                            if len(section.text) > 0:
                                    sections.extend([section])
                            # recursive call for each section
                            for s in section.sections:
                                    tmp = self.__flattenSections(s)
                                    if tmp is not None:
                                            sections.extend(tmp)
                            return sections
            else: 
                    return None