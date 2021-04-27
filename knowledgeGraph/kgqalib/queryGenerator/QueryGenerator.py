import networkx as nx
import pandas as pd
import sparql
import logging
from tabulate import tabulate
import nltk
import re
nltk.download('wordnet')
from nltk.corpus import wordnet

class QueryGenerator():

    def generate(self, question, graph):
        # query head and tail
        variable = self.__get_target_variable(graph)
        (head, tail) = self.__generate_head_and_tail(question, variable)
        # query body
        body = self.__generate_body(graph)
        # assemble query
        query = head + body + tail
        return query
    
    def ask(self, query, endpoint='http://dbpedia.org/sparql'):
        logging.debug("Ask query:")
        logging.debug(query)

        try:
            results=sparql.query(endpoint, query)
        except Exception as e:
            logging.error(f"Exception {e} on query:\n\n{query}")
            raise e

        return pd.DataFrame([[item.n3() for item in row] for row in results], columns=['Answers'])
    
    def generate_and_ask(self, question, graph, endpoint='http://dbpedia.org/sparql'):
        query = self.generate(question, graph)

        return self.ask(query, endpoint)


    def __generate_triple(self, Q, edge):
        # src node
        subj = Q.nodes[edge[0]]['label']
        # edge
        pred = edge[2]['label']
        # dst node
        obj = Q.nodes[edge[1]]['label']
        # generate triple
        triple = ' '.join([subj, pred, obj, '.', '\n'])
        return triple

    def __generate_body(self, graph):
        triples = []
        # generate the triple for each edge
        for edge in graph.edges.data():
            triples.append(self.__generate_triple(graph, edge))
        # join triples into the body
        body = '{\n' + ' '.join(triples) + '}'
        return body

    def __get_target_variable(self, graph):
        # !!! emergency variable, hope it's not needed !!!
        candidates = []
        # search for variables in nodes
        for node in graph.nodes.data():
            node_id = node[0]
            node_label = node[1]['label']
            # keep only the variable nodes
            if node_label.startswith('?'):
                # !!! emergency operation, hope it's not needed !!!
                candidates.append(node_label)
                # the target variable should be a non-intermediate node
                if graph.out_degree(node_id) == 0 or graph.in_degree(node_id) == 0:
                    return node_label
        
        # if we are here I made a wrong assumption lol
        return ' '.join(candidates)

    # from ordinal number computes synonyms
    def convert_ordinal_number(list_ordinal_num):
        list_ordinal_syns=[]
        for elem in list_ordinal_num:
            list_ordinal_syns.append(wordnet.synsets(elem)[0].lemma_names())
        return list_ordinal_syns

    # compute first n ordinal numbers
    def compute_ordinal_number(n=10):
        ordinal = lambda num: "%d%s" % (num,"tsnrhtdd"[(num//10%10!=1)*(num%10<4)*num%10::4])
        list_ordinal_number=[ordinal(i) for i in range(1,n)]
        return convert_ordinal_number(list_ordinal_number)
    
    # check if the word is an ordinal number
    def is_ordinal_number(word):
        ordinal_numbers = compute_ordinal_number()
        for elem in ordinal_numbers:
            if word in elem:
                return (True,elem)
        return (False,[])
    
    # check if a list of tokens contains an ordinal number and return the corresponding value for the limit
    def contain_ordinal_number(tokens):
        for token in tokens:
            check, number = is_ordinal_number(token)
            if check:
                return number[1][:-2]
        return []
    
    # computes contraints for the question
    def constraint(question):
        category=[]
        
        question= re.sub(' +', ' ', question)
        if "how many" in question or "number of" in question or "count of" in question:
            category.append("aggregation") # count
        
        tokens = nltk.word_tokenize(question)
        tags = [ elem[1] for elem in nltk.pos_tag(tokens)]
        print(tags)
        if "VB" in tags[0]:
            category.append("answer-type") # ask
        
        '''ordinal_number = contain_ordinal_number(tokens)
        if len(ordinal_number)>0:  
            category.append("ordinal") # limit'''
        return category, ordinal_number

    # TODO: put here gabri's code to distinguish between different question types
    def __generate_head_and_tail(self, question, variable):
        print("Question: ", question)
        constraints, ordinal = constraint(question)
        print("Constraints: ", constraints)
        print("Ordinal: ", ordinal)
        if "answer-type" in constraints:
            head = "ASK "
        elif "aggregation" in constraints:
            head = "SELECT (count(distinct "+ variable +") as ?count) "
        else:
            head = "SELECT DISTINCT " + variable + " WHERE "
        
        '''if "ordinal" and type(ordinal) == str:
            tail= "limit " + ordinal
        else:'''
        tail = ""
        #tail = "\n"
        return (head, tail)



    
   

    