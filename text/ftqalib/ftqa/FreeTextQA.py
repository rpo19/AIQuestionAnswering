from transformers import pipeline
import wikipediaapi
import wikipedia as wiki
from operator import itemgetter


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

    def answerFromWiki(self, question, entity=None, top=0, debug=False): 
        p = None
        if entity:
                # extract resource name from entity
                entity = entity.split('/')[-1].replace('>','')
                # get wikipedia page
                p = self.__wiki.page(entity)
        else:
                # search page by question
                results = wiki.search(question)
                # get wikipedia page
                p = self.__wiki.page(results[0])

        # get flattened sections of the page
        sections = self.__flattenSections(p)
        print('Number of sections:', len(sections))
        # iterate on sections and find answer in each span
        answers = []
        print('Answering from sections...')
        for section in sections:
                context = section.text
                if context != '':
                        answer = self.answerFromSpan(question=question, context=context)
                        answer['entity'] = entity
                        answer['section'] = section.title
                        answers.append(answer)
                        if debug:
                                tokens = self.__qa.tokenizer(context)['input_ids']
                                print(section.title, f'(length: {len(tokens)}')
                                print(f"Answer: '{answer['answer']}' with score {answer['score']}", '\n', 30*'_')
        # order answers by descending score
        ordered_answers = sorted(answers, key=itemgetter('score'), reverse=True)
        # get only the top n
        if top > 0:
                ordered_answers = ordered_answers[:top]
        return ordered_answers

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