# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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
import generateQuery
import itertools
import time


class QueryGraphBuilder():
    def __init__(self, embeddings=None, bert_similarity=True, entities_cap=25):
        self.entities_cap = entities_cap
        if bert_similarity:
            self.vectorizer = Vectorizer()
        else:
            if embeddings is None:
                raise Exception("Embeggins not found.")
            else:
                self.embeddings = embeddings
        self.var_num = 0
        self.stops = stopwords.words('english')
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
        self.bert_similarity = bert_similarity
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

        try:
            results=sparql.query(endpoint, query)
        except Exception as e:
            print(f"Exception {e} on query:\n\n{query}")
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
        print(f"{desc} Elapsed time: {elapsed}") 
        return ret

    """
    Build query graph by a single step.

    :param question: natural language question
    :param cn: cn entity resources
    :param Q: query graph
    :param NS: NS

    :return: query graph
    """
    def build_step(self, question, cn, Q, NS, entities):
        # get relations connected to cn
        R = self.__get_time("Get relations", self.__get_relations, (Q, NS, cn))
        print(f"Got relations: {R.shape}")

        if R is None:
            # leaf node
            if len(NS) > 1:
                raise Exception(f"""
                    NS should contain only one element.
                    NS: {NS}
                    """)
            unlabeled_node=NS[0]
            if cn in entities:
                Q.nodes[unlabeled_node]['label']=cn
                entities.remove(cn)
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

        r=self.__get_time("Get most relevant", self.__get_most_relevant_relation, (question, R))

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
            return Q, [], []

        # assemble relation in Q
        Q[unlabeled_edge[0]][unlabeled_edge[1]]['label']=r["pred"]
        # for drawing purposes
        Q[unlabeled_edge[0]][unlabeled_edge[1]]['short_label']=r["label"]

        self.var_num += 1

        # get adjacent unexplored node
        NS=self.__get_adjacent_unexplored(Q)
        # get entities corresponding to NS
        cn=self.__get_time("Get candidates", self.__get_candidate_entities, (Q, NS, entities))

        return Q, NS, cn

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
    Build query graph.

    :param question: natural language question
    :param entity: entity resource
    :param pattern: graph pattern of the question

    :return: query graph
    """
    def build(self, question, entities, pattern):

        question, cn, Q, NS=self.pre_build(question, entities, pattern)

        while NS:
            Q, NS, cn=self.build_step(question, cn, Q, NS, entities)

        return Q

    def ask_step(self, question, Q, endpoint='http://dbpedia.org/sparql'):
        query=generateQuery.generateQuery(question, Q)

        try:
            results=sparql.query(endpoint, query)
        except Exception as e:
            print(f"Exception {e} on query:\n\n{query}")
            raise e

        return pd.DataFrame([[item.n3() for item in row] for row in results])

    def ask(self, question, entities, pattern, endpoint='http://dbpedia.org/sparql'):
        Q=self.build(question, entities, pattern)

        return self.ask_step(question, Q, endpoint)

    """
    Get graph pattern for a pattern p.

    :param pattern: pattern dictionary

    :return: networkx graph of the pattern p
    """
    def __get_pattern(self, pattern):
        return nx.from_dict_of_lists(self.patterns[pattern],
                                     create_using=nx.DiGraph)

    """
    Get non intermediate notes for a graph pattern p.

    :param p: graph pattern

    :return: dict of non-intermediary nodes
    """
    def __get_non_intermediate_nodes(self, p):
        return [node for node in p.nodes if p.out_degree(node) + p.in_degree(node) < 2]

    """
    Get outgoing or incoming relations for an entity.

    :param entity: entity for which you want to find relations
    :param query_type: 'outgoing' for outgoing relations, 'incoming' for incoming relations
    :param query_type: SPARQL endpoint

    :return: dataframe of outgoing/incoming relations (URI, label)
    """
    def __get_relations(self, Q, NS, cn, endpoint='http://dbpedia.org/sparql'):

        # i dont care about ingoing and outgoing as super query is already super

        if not cn:
            return None

        if not NS:
            raise Exception("NS should not be empty!")

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

        try:
            results=sparql.query(endpoint, query)
        except Exception as e:
            print(f"Exception {e} on query:\n\n{query}")
            raise e

        def results_generator(res):
            # for sub,pred,obj,label,given in res:
            for pred, label, given, entity in res:
                yield {
                    # "sub": sub.n3(),
                    "pred": pred.n3(),
                    # "obj": obj.n3(),
                    "label": label.value.replace('-', ' ').lower()
                        if label else self.__parse_predicate(pred),
                    "given": given.value,
                    "entity": entity.n3()
                    }

        triples=pd.DataFrame(results_generator(results))

        return triples


    """
    Parse URI to extract a label.

    :param pred: predicate URI

    :return: predicate label
    """
    def __parse_predicate(self, pred):
        last=pred.rsplit('/', 1)[1]
        splitted=re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', last)
        return ' '.join(splitted)

    """
    Get most relevant relation using given embedding and levenshtein_distance.

    :param question: question in natural language
    :param R: set of candidate relations
    :param lambda_param: hyperparameter describing the importance of cosine similarity and levenshtein_distance

    :return: label of most relevant relation
    """
    def __get_most_relevant_relation(self, question, R, lambda_param=0.5):
        unique_relations=R
        question=question.lower().replace('?', ' ?')
        # tokenize question
        question_tokens=question.split()
        # remove stopwords and punctuation tokens
        question_tokens=[token for token in question_tokens
                               if token not in self.stops and token not in string.punctuation]

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
                    lev_distance=levenshtein_distance(
                        question_token, rel_token)
                    # sum to previous relenvances of relation tokens and question tokens
                    relevance += lambda_param * cos_sim + \
                        (1 - lambda_param) * 1/(lev_distance+1)

            relevances.append(relevance/len(relation_tokens))
        relevances=np.array(relevances)

        return unique_relations.iloc[np.argmax(relevances)]


if __name__ == "__main__":

    import time
    import pickle
    start=time.time()
    embeddings=pickle.load(
        open('../../data/glove.twitter.27B.200d.pickle', 'rb'))
    stop=time.time()
    n=stop - start
    print(f"Loaded embeddings in {n} seconds.")

    query_builder=QueryGraphBuilder(
        embeddings=embeddings, bert_similarity=False)

    res=query_builder.ask(question='What university campuses are situated in Indiana?',
                            entities=["<http://dbpedia.org/resource/Indiana>"],
                            pattern='p2')

    print(res)
