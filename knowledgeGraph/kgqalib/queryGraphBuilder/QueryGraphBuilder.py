import networkx as nx
import sparql
import pandas as pd
import re
import numpy as np
import string
from numpy import dot
from numpy.linalg import norm
from Levenshtein import distance as levenshtein_distance
from nltk.corpus import stopwords
from scipy import spatial
import itertools
import time
import logging
import pickle
import flair
import math
from pathlib import Path
flair.cache_root = Path('./data/flair')
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
import sys


from .MmapWordEmbeddings import MmapWordEmbeddings


class QueryGraphBuilder():
    def __init__(self, embeddings=None, path_to_embeddings='../../data/glove.twitter.27B.200d.pickle',
                        entities_cap=25, mode='glove'):
        self.entities_cap = entities_cap
        self.mode = mode
        if embeddings is None:
            print('Loading embeddings...')
            if mode == 'glove':
                try:
                    # glove 840B-300d
                    if sys.platform == 'linux':
                        # pro linux users use mmap to save RAM
                        self.embeddings= MmapWordEmbeddings(path_to_embeddings)
                    else:
                        self.embeddings= WordEmbeddings(path_to_embeddings)

                except Exception as e:
                    raise e
            elif mode == 'sentence_roberta':
                self.embeddings = SentenceTransformerDocumentEmbeddings('stsb-roberta-large')
            elif mode == 'stacked':
                # init standard GloVe embedding
                glove_embedding = WordEmbeddings('glove')

                # init Flair forward and backwards embeddings
                flair_embedding_forward = FlairEmbeddings('news-forward')
                flair_embedding_backward = FlairEmbeddings('news-backward')
                self.embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
                                       ])
        else:
            self.embeddings = embeddings
        self.var_num = 0
        self.stops = self.__get_stopwords()
        self.exclusions_list = [
            '<http://dbpedia.org/property/wikiPageUsesTemplate>',
            '<http://dbpedia.org/ontology/wikiPageExternalLink>',
            '<http://dbpedia.org/ontology/wikiPageID>',
            '<http://dbpedia.org/ontology/wikiPageRevisionID>',
            '<http://dbpedia.org/ontology/wikiPageLength>',
            '<http://dbpedia.org/ontology/wikiPageWikiLink>',
            '<http://www.w3.org/2000/01/rdf-schema#label>',
            '<http://www.w3.org/2002/07/owl#sameAs>',
            '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>',
            '<http://schema.org/sameAs>',
            '<http://purl.org/dc/terms/subject>',
            '<http://xmlns.com/foaf/0.1/isPrimaryTopicOf>',
            '<http://xmlns.com/foaf/0.1/depiction>',
            '<http://www.w3.org/2000/01/rdf-schema#seeAlso>',
            '<http://www.w3.org/2000/01/rdf-schema#comment>',
            '<http://dbpedia.org/ontology/abstract>',
            '<http://dbpedia.org/ontology/thumbnail>',
            '<http://dbpedia.org/property/caption>',
            '<http://dbpedia.org/property/captionAlign>',
            '<http://dbpedia.org/property/image>',
            '<http://dbpedia.org/property/imageFlag>',
            '<http://www.w3.org/ns/prov#wasDerivedFrom>',
            '<http://dbpedia.org/ontology/wikiPageRedirects>',
            '<http://dbpedia.org/ontology/wikiPageDisambiguates>',
            '<http://dbpedia.org/property/1namedata>'
        ]
        self.patterns = {
            'p0': {'A': []},
            'p1': {'A': ['B']},
            'p2': {'A': ['B'],
                'B': ['C']},
            'p3': {'A': ['B'],
                'C': ['B']},
            'p4': {'A': ['B', 'C']},
            'p5': {'A': ['B', 'C', 'D']},
            'p6': {'A': ['B', 'C'],
                'C': ['D']},
            'p7': {'A': ['B'],
                'B': ['C'],
                'C': ['D']},
            'p8': {'A': ['B'],
                'B': ['C'],
                'D': ['C']},
            'p9': {'A': ['B'],
                'B': ['C'],
                'D': ['B']},
            'p10': {'A': ['B'],
                'B': ['C', 'D']},
            'p11': {'A': ['B'],
                'C': ['B'],
                'D': ['B']}
        }

    def __get_stopwords(self):
        stops = stopwords.words('english')
        stops.remove('where')
        stops.remove('which')
        stops.remove('what')
        stops.remove('when')
        return stops

    def __get_unlabeled_edge(self, graph_pattern, node):
        return next(
                (edge[0:2] for edge in itertools.chain(
                    graph_pattern.in_edges(node, data=True),
                    graph_pattern.out_edges(node, data=True))
                    if not edge[2]
                ),
            None)

    def __get_adjacent_unexplored(self, Q):
        # get only one
        neighbors=set()
        for node in Q.nodes:
            if Q.nodes[node]:
                for neighbor in nx.all_neighbors(Q, node):
                    if not Q.nodes[neighbor]:
                        return [neighbor]
        return []

    def __get_candidate_entities(self, Q, NS, entities, endpoint='http://dbpedia.org/sparql'):
        if not NS:
            return []
        def get_node_label(node, NS):
            found=Q.nodes[node]
            if found:
                return found["label"]
            else:
                if node in NS:
                    return "?var"
                else:
                    return "?"+node.lower()

        def get_pred_label(pred, sub, obj):
            return "?"+sub.lower()+obj.lower()

        current_triples=[(get_node_label(sub, NS), pred["label"], get_node_label(obj, NS))
                                for sub, obj, pred in Q.edges(data=True) if "label" in pred]

        current_triples.extend([(get_node_label(sub, NS), get_pred_label(pred, sub, obj), get_node_label(obj, NS))
                        for sub, obj, pred in Q.edges(data=True) if "label" not in pred])

        body="\n    ".join(
            [f"{sub} {pred} {obj} ." for sub, pred, obj in current_triples])
        query=f"""
SELECT DISTINCT ?var
WHERE
{{
    {body}
}}
"""

        logging.debug("Candidates query:")
        logging.debug(query)

        try:
            results=sparql.query(endpoint, query)
        except Exception as e:
            logging.error(f"Exception {e} on query:\n\n{query}")
            raise e

        result_list=[]
        for var in results:
            n3=var[0].n3()
            if n3 in entities:
                result_list.insert(0, n3)
            else:
                result_list.append(n3)

        return result_list[:self.entities_cap]

    def __get_time(self, desc, function, args):
        start = time.time()
        ret = function(*args)
        elapsed = time.time() - start
        logging.debug(f"{desc} Elapsed time: {elapsed}")
        return ret

    def __are_there_unlabeled_edges(self, Q):
        # None when all labeled
        res = next((edge for edge in Q.edges(data=True) if not edge[2]), None)
        return res is not None

    def __get_intersection(self, cn, entities):
        cn_set = set(cn)
        entities_set = set(entities)
        return list(cn_set.intersection(entities_set))

    """
    Build query graph.
    """
    def build(self, question, entities, entities_texts, pattern):
        logging.debug("Building graph...")
        logging.debug(f"question: {question}")
        logging.debug(f"entities: {entities}")
        logging.debug(f"pattern: {pattern}")

        question, cn, Q, NS=self.pre_build(question, entities, pattern)

        while NS:
            Q, NS, cn=self.build_step(question, cn, Q, NS, entities, entities_texts)

        return Q

    """
    Prepare graph builder.
    """
    def pre_build(self, question, entities, pattern):
        # TODO: higher score linking
        cn=[entities[0]]
        # get pattern graph
        p=self.__get_pattern(pattern)
        # make a copy of the pattern
        Q=p.copy()
        # get non-intermediate nodes
        NS=self.__get_non_intermediate_nodes(p)

        self.var_num=0

        return question, cn, Q, NS

    """
    Build query graph by a single step.
    """
    def build_step(self, question, cn, Q, NS, entities, entities_text):

        exists_unlabeled_edge = self.__are_there_unlabeled_edges(Q)

        if not exists_unlabeled_edge:
            # leaf node
            if len(NS) > 1:
                raise Exception(f"""
                    NS should contain only one element.
                    NS: {NS}
                    """)
            unlabeled_node=NS[0]

            intersection_cn_entities = self.__get_intersection(cn, entities)

            if len(intersection_cn_entities) > 1:
                raise Exception(f"""
                    Intersection should give only 1 value.
                    intersection_cn_entities: {intersection_cn_entities}
                    """)

            if intersection_cn_entities:
                Q.nodes[unlabeled_node]['label']= intersection_cn_entities[0]
                entities.remove(intersection_cn_entities[0])
            elif self.var_num > 0:
                # variable
                Q.nodes[unlabeled_node]['label']='?' + str(self.var_num)
            else:
                raise Exception("Cannot find first entity")

            # get adjacent unexplored node
            NS=self.__get_adjacent_unexplored(Q)
            # get entities corresponding to NS
            cn=self.__get_candidate_entities(Q, NS, entities)

            return Q, NS, cn

        # get relations connected to cn
        R = self.__get_time("Get relations", self.__get_relations, (Q, NS, cn))
        logging.debug(f"Got relations: {R.shape}")

        if R.shape[0] == 0:
            logging.debug('No relations found. Aborting...')
            return None, [], None

        r, r_top_10 = self.__get_time("Get most relevant", self.__get_most_relevant_relation, (question, R, entities_text))

        # get an unlabelled node in NS which has a relation respecting r direction
        unlabeled_node=r["given"]

        # first time has to be entity
        if r["entity"] in entities:
            # entity found
            Q.nodes[unlabeled_node]['label']=r["entity"]
            entities.remove(r["entity"])
        elif self.var_num > 0:
            # variable
            Q.nodes[unlabeled_node]['label']='?' + str(self.var_num)
        else:
            raise Exception("Cannot find first entity")

        # get unlabeled edge from unlabeled_node
        unlabeled_edge=self.__get_unlabeled_edge(Q, unlabeled_node)
        if unlabeled_edge is None:
            raise Exception("No unlabeled edge. This should not happen!")

        # assemble relation in Q
        Q[unlabeled_edge[0]][unlabeled_edge[1]]['label']=r["pred"]
        # for drawing purposes
        Q[unlabeled_edge[0]][unlabeled_edge[1]]['short_label']=r["label"]
        # top 10 relations
        Q[unlabeled_edge[0]][unlabeled_edge[1]]['top_10'] = r_top_10.to_dict('records')

        self.var_num += 1

        # get adjacent unexplored node
        NS=self.__get_adjacent_unexplored(Q)
        # get entities corresponding to NS
        cn=self.__get_time("Get candidates", self.__get_candidate_entities, (Q, NS, entities))

        return Q, NS, cn

    def ask(self, question, entities, pattern, endpoint='http://dbpedia.org/sparql'):
        Q=self.build(question, entities, pattern)

        return self.ask_step(question, Q, endpoint)

    def ask_step(self, question, Q, endpoint='http://dbpedia.org/sparql'):
        query=generateQuery.generateQuery(question, Q)

        logging.debug("Ask query:")
        logging.debug(query)

        try:
            results=sparql.query(endpoint, query)
        except Exception as e:
            logging.error(f"Exception {e} on query:\n\n{query}")
            raise e

        return pd.DataFrame([[item.n3() for item in row] for row in results])



    """
    Get graph pattern for a pattern p.
    """
    def __get_pattern(self, pattern):
        return nx.from_dict_of_lists(self.patterns[pattern],
                                     create_using=nx.DiGraph)

    """
    Get non intermediate notes for a graph pattern p.
    """
    def __get_non_intermediate_nodes(self, p):
        return [node for node in p.nodes if p.out_degree(node) + p.in_degree(node) < 2]

    """
    Get outgoing or incoming relations for an entity.
    """
    def __get_relations(self, Q, NS, cn, endpoint='http://dbpedia.org/sparql'):

        # i dont care about ingoing and outgoing as super query is already super

        if not cn:
            return None

        if not NS:
            raise Exception("NS should not be empty!")

        logging.debug("Getting relations starting from:")
        logging.debug(f"NS: {len(NS)}")
        logging.debug(f"cn: {len(cn)}")

        # computationals: |NS| * |cn|

        # get the label of ns given permutation
        def get_node_label(node, NS, permutation, Q):
            found=Q.nodes[node]
            # already labeled (not in ns) # check it?
            if found:
                return found["label"]

            # if node in NS:
            #     index = NS.index(node)
            #     entity = permutation[index]

            #     if entity:
            #         return entity

            # if not in NS or entity is None
            return "?"+node.lower()

        def get_pred_label(pred, sub, obj, NS, permutation, Q):
            if sub == permutation[0] or obj == permutation[0]:
                return "?pred"

            return "?"+sub.lower()+obj.lower()

        exclusions_string=",\n                ".join(self.exclusions_list)

        def triples_generator():
            for permutation in set(itertools.product(NS, cn)):

                current_triples=[
                        (
                            get_node_label(sub, NS, permutation, Q),
                            pred["label"],
                            get_node_label(obj, NS, permutation, Q)
                        ) for sub, obj, pred in Q.edges(data=True) if "label" in pred
                    ]

                current_triples.extend([
                        (
                            get_node_label(sub, NS, permutation, Q),
                            get_pred_label(pred, sub, obj, NS, permutation, Q),
                            get_node_label(obj, NS, permutation, Q)
                        ) for sub, obj, pred in Q.edges(data=True) if "label" not in pred
                    ])

                body="\n        ".join(
                    [f"{sub} {pred} {obj} ." for sub, pred, obj in current_triples])

                body += f"""
        VALUES (?{permutation[0].lower()} ?given ?entity) {{
            ({permutation[1]} "{permutation[0]}" {permutation[1]})
        }} .
        FILTER (
            lang (?pred_label) = 'en'
        ) .
        OPTIONAL {{
            ?pred rdfs:label ?pred_label .
            BIND (
                STR(?pred_label) AS ?pred_label_stripped
            ) .
        }} .
        FILTER (
            ?pred NOT IN (
                {exclusions_string}
            )
        ) .
                """
                yield body

        body="""
    }
    UNION
    {
        """.join(list(triples_generator()))

        query=f"""
SELECT DISTINCT ?pred ?pred_label ?given ?entity
WHERE
{{
    {{
        {body}
    }}
}}
"""

        logging.debug("Relations query:")
        logging.debug(query)

        query_success = False
        tries = 0

        while not query_success:
            try:
                results=sparql.query(endpoint, query, timeout=20)
                query_success = True
                
            except Exception as e:
                query_success = False
                print('Timeout getting relations, retrying...')
                if tries == 7:
                    raise e
                else:
                    tries += 1
                #logging.error(f"Exception {e} on query:\n\n{query}")
                #raise e
                
            



        # def results_generator(res):
        #     # for sub,pred,obj,label,given in res:
        #     for pred, label, given, entity in res:
        #         yield {
        #             # "sub": sub.n3(),
        #             "pred": pred.n3(),
        #             # "obj": obj.n3(),
        #             "label": label.value.replace('-', ' ').lower()
        #                 if label else self.__parse_predicate(pred),
        #             "given": given.value,
        #             "entity": entity.n3()
        #             }
        def generate_results(res):
            triples = pd.DataFrame(columns=['pred', 'label', 'given', 'entity'])
            for pred, label, given, entity in res:

                if label:
                    label = label.value.replace('-', ' ').lower()
                else:
                    label = self.__parse_predicate(pred.value)

                tmp = triples[triples.label == label]


                if len(tmp) > 1:
                    raise Exception("There should be only 1 duplicate.")

                triple = {
                            "pred": pred.n3(),
                            "label": label,
                            "given": given.value,
                            "entity": entity.n3()
                         }
                if tmp.empty:
                    triples = triples.append(triple, ignore_index=True)
                else:
                    if "http://dbpedia.org/ontology/" in pred.value:
                        triples.iloc[tmp.index.values[0]] = triple

            return triples

        triples = generate_results(results)

        return triples


    """
    Parse URI to extract a label.
    """
    def __parse_predicate(self, pred):
        last=pred.rsplit('/', 1)[1]
        splitted=re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', last)
        return ' '.join(splitted)

    """
    Get most relevant relation.
    """
    def __get_most_relevant_relation(self, question, R, entities_texts, lambda_param=0.55):
        if self.mode == 'glove':
            return self.__get_most_relevant_relation_glove_levenshtein(question, R, entities_texts, lambda_param)
        elif self.mode == 'sentence_roberta':
            return self.__get_most_relevant_relation_sentence_roberta(question, R, entities_texts, lambda_param)
        elif self.mode == 'stacked':
            return self.__get_most_relevant_relation_stacked_embeddings(question, R, entities_texts, lambda_param)

    """
    Get most relevant relation using given embedding and levenshtein_distance.
    """
    def __get_most_relevant_relation_glove_levenshtein(self, question, R, entities_texts, lambda_param):
        if R.shape[0] == 0:
            logging.debug('__get_most_relevant_relation_glove_levenshtein no relations.')
            return None, []

        unique_relations=R

        # preprocess question
        question = self.__preprocess_question(question, entities_texts)

        relevances=[]

        flair_question = Sentence(question)
        # embed the sentence
        self.embeddings.embed(flair_question)

        for index, row in unique_relations.iterrows():
            flair_relation = Sentence(row['label'])
            self.embeddings.embed(flair_relation)
            relevance=0
            for rel_token in flair_relation:
                for question_token in flair_question:
                    rel_token_embedding = rel_token.embedding.tolist()
                    question_token_embedding = question_token.embedding.tolist()


                    cos_sim = 1 - spatial.distance.cosine(rel_token_embedding, question_token_embedding)
                    if math.isnan(cos_sim):
                        cos_sim = 0

                    length = max(len(question_token.text), len(rel_token.text))
                    lev_sim = (length - levenshtein_distance(question_token.text, rel_token.text)) / length
                    # sum to previous relenvances of relation tokens and question tokens
                    relevance += lambda_param * cos_sim + (1 - lambda_param) * lev_sim

            relevances.append(relevance/len(flair_relation.tokens))
        relevances=np.array(relevances)


        # get top 10 relations
        top_10_relations = self.__get_top_relevants(R, relevances)
        return top_10_relations.iloc[0], top_10_relations[['pred', 'relevance']].iloc[0:10]

    """
    Get most relevant relation using stacked word level embeddings
    """
    def __get_most_relevant_relation_stacked_embeddings(self, question, R, entities_texts, lambda_param):
        unique_relations=R

        relevances=[]

        flair_question = Sentence(question)
        # embed the sentence
        self.embeddings.embed(flair_question)

        for index, row in unique_relations.iterrows():
            flair_relation = Sentence(row['label'])
            self.embeddings.embed(flair_relation)
            relevance=0
            for rel_token in flair_relation:
                for question_token in flair_question:
                    cos_sim = 1 - spatial.distance.cosine(rel_token.embedding.tolist(), question_token.embedding.tolist())

                    length = max(len(question_token.text), len(rel_token.text))
                    lev_sim = (length - levenshtein_distance(question_token.text, rel_token.text)) / length
                    # sum to previous relenvances of relation tokens and question tokens
                    relevance += lambda_param * cos_sim + (1 - lambda_param) * lev_sim

            relevances.append(relevance/len(flair_relation.tokens))
        relevances=np.array(relevances)

        # get top 10 relations
        top_10_relations = self.__get_top_relevants(R, relevances)

        return top_10_relations.iloc[0], top_10_relations[['pred', 'relevance']].iloc[0:10]

    """
    Get most relevant relation using sentence level embedding with Roberta.
    """
    def __get_most_relevant_relation_sentence_roberta(self, question, R, entities_texts, lambda_param):
        unique_relations=R

        relevances=[]

        flair_question = Sentence(question)
        # embed the sentence
        self.embeddings.embed(flair_question)

        for index, row in unique_relations.iterrows():
            print('label: {}'.format(row['label']))
            flair_relation = Sentence(row['label'])
            self.embeddings.embed(flair_relation)

            relevance = 1 - spatial.distance.cosine(flair_question.embedding.tolist(), flair_relation.embedding.tolist())
            relevances.append(relevance)

        relevances=np.array(relevances)

        # get top 10 relations
        top_10_relations = self.__get_top_relevants(R, relevances)

        return top_10_relations.iloc[0], top_10_relations[['pred', 'relevance']].iloc[0:10]

    """
    Get most relevant relation using given embedding and levenshtein_distance // DEPRECATED.
    """
    def __get_most_relevant_relation_glove_levenshtein_old(self, question, R, entities_texts, lambda_param):
        unique_relations=R

        # preprocess question
        question = self.__preprocess_question(question, entities_texts)

        relevances=[]

        for index, row in unique_relations.iterrows():
            # tokenize label
            relation_tokens=row['label'].split()

            relevance=0
            for rel_token in relation_tokens:

                for question_token in question_tokens:

                    if rel_token in self.embeddings and question_token in self.embeddings:

                        rel_token_embedding=self.embeddings[rel_token]
                        question_token_embedding=self.embeddings[question_token]

                        # compute cosine similarity
                        cos_sim=1 - \
                            spatial.distance.cosine(
                                rel_token_embedding, question_token_embedding)
                    else:
                        cos_sim=0
                    # compute lev distance
                    length = max(len(question_token), len(rel_token))
                    lev_sim = (length - levenshtein_distance(question_token, rel_token)) / length
                    # sum to previous relenvances of relation tokens and question tokens
                    relevance += lambda_param * cos_sim + \
                        (1 - lambda_param) * lev_sim

            relevances.append(relevance/len(relation_tokens))
        relevances=np.array(relevances)

        # get top 10 relations
        top_10_relations = self.__get_top_relevants(R, relevances)

        return top_10_relations.iloc[0], top_10_relations[['pred', 'relevance']].iloc[0:10]

    def __preprocess_question(self, question, entities_texts):

        if entities_texts:
            # remove entities
            for entity_text in entities_texts:
                question = question.replace(entity_text, ' ')

        question = question.lower().replace('?', ' ?')

        question = question.replace('where', 'place')
        question = question.replace('when', 'date')

        # tokenize question
        question_tokens=question.split()
        # remove stopwords and punctuation tokens
        question_tokens=[token for token in question_tokens
                               if token not in self.stops and token not in string.punctuation]
        return ' '.join(question_tokens)

    def __get_top_relevants(self,R, relevances):
        tmp = R.copy()
        tmp['relevance'] = relevances
        tmp = tmp.sort_values(by='relevance', ascending=False)
        return tmp


if __name__ == "__main__":

    import time
    import pickle

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    start=time.time()
    embeddings=pickle.load(
        open('../../data/glove.twitter.27B.200d.pickle', 'rb'))
    stop=time.time()
    n=stop - start
    logging.debug(f"Loaded embeddings in {n} seconds.")

    query_builder=QueryGraphBuilder(
        embeddings=embeddings)

    res=query_builder.ask(question='What university campuses are situated in Indiana?',
                            entities=["<http://dbpedia.org/resource/Indiana>"],
                            pattern='p2')

    print(res)
