import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from knowledgeGraph.kgqalib.dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
from knowledgeGraph.kgqalib.patternBuilder.SQPBuilder import SQPBuilder
from knowledgeGraph.kgqalib.queryGraphBuilder.QueryGraphBuilder import QueryGraphBuilder
from knowledgeGraph.kgqalib.queryGenerator.QueryGenerator import QueryGenerator
import knowledgeGraph.kgqalib.shared.utils as utils

def load_models():
    MODEL_PATH = './data/models/pattern_classifier.h5'
    EMBEDDINGS_PATH = './data/glove.twitter.27B.200d.pickle'

    # instantiate modules
    ## Pattern classifier
    pattern_classifier = SQPBuilder().load(MODEL_PATH)

    ## Entity extractor
    entity_extractor = DBPediaEntityExtractor()

    ## Query graph builder
    query_graph_builder = QueryGraphBuilder(path_to_embeddings = EMBEDDINGS_PATH, bert_similarity=False)

    ## Query generator
    query_generator = QueryGenerator()

    return pattern_classifier, entity_extractor, query_graph_builder, query_generator