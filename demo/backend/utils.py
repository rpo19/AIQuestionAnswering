import os
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from knowledgeGraph.kgqalib.dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
from knowledgeGraph.kgqalib.patternBuilder.SQPBuilder import SQPBuilder
from knowledgeGraph.kgqalib.queryGraphBuilder.QueryGraphBuilder import QueryGraphBuilder
from knowledgeGraph.kgqalib.queryGenerator.QueryGenerator import QueryGenerator
from text.ftqalib.ftqa.FreeTextQA import FreeTextQA
import knowledgeGraph.kgqalib.shared.utils as utils

def load_models_kgqa():
    MODEL_PATH = './data/models/pattern_classifier.h5'
    EMBEDDINGS_PATH = './data/glove.840B.300d.gensim'
    GRAPH_BUILDER_MODE = 'glove' # 'glove' | 'stacked' | 'sentence_roberta'
    NER_MODE = 'spotlight' # 'spotlight' | 'custom'
    ENTITIES_CAP = 50


    # instantiate modules
    ## Pattern classifier
    pattern_classifier = SQPBuilder().load(MODEL_PATH)

    ## Entity extractor
    entity_extractor = DBPediaEntityExtractor(mode=NER_MODE)

    ## Query graph builder
    query_graph_builder = QueryGraphBuilder(mode=GRAPH_BUILDER_MODE, path_to_embeddings=EMBEDDINGS_PATH, entities_cap=ENTITIES_CAP)

    ## Query generator
    query_generator = QueryGenerator()

    return pattern_classifier, entity_extractor, query_graph_builder, query_generator

def load_models_ftqa():
    free_text_answerer = FreeTextQA()
    return free_text_answerer

def to_dict_of_dicts(Q):
    edge_list = nx.to_dict_of_lists(Q)
    nodes = []
    edges = []

    for index, start_node in enumerate(edge_list):
        node = {
            'id': start_node,
            'label': Q.nodes[start_node]['label']
        }
        nodes.append(node)

        for end_node in edge_list[start_node]:
            edge = {
                'id': start_node + '_' + end_node,
                'start': start_node,
                'end': end_node,
                'label': Q[start_node][end_node]['label'],
                'top_10': Q[start_node][end_node]['top_10']
            }
            edges.append(edge)
    
    return {
        'nodes': nodes,
        'edges': edges
    }

    