import networkx as nx
import pandas as pd
import sparql
import logging
from tabulate import tabulate

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

    # TODO: put here gabri's code to distinguish between different question types
    def __generate_head_and_tail(self, question, variable):
        head = "SELECT DISTINCT " + variable + " WHERE "
        tail = ""
        #tail = "\n"
        return (head, tail)



    
   

    