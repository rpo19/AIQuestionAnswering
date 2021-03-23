import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
from patternBuilder.SQPBuilder import SQPBuilder
from queryGraphBuilder.QueryGraphBuilder import QueryGraphBuilder
from queryGenerator.QueryGenerator import QueryGenerator
import shared.utils as utils
import pandas as pd


def ask(question):
    # classify question with pattern
    patterns, raw_probs = pattern_classifier.transform(question)
    # extract and link entities
    entities, texts = entity_extractor.extract(question)
    # query graph construction
    Q = query_graph_builder.build(question, entities, patterns[0])
    # build SPARQL query and retrieve answers
    answers_df = query_generator.generate_and_ask(question, Q)
    return answers_df


MODEL_PATH = '../../data/models/pattern_classifier.h5'

# instantiate modules
## Pattern classifier
pattern_classifier = SQPBuilder().load(MODEL_PATH)

## Entity extractor
entity_extractor = DBPediaEntityExtractor()

## Query graph builder
query_graph_builder = QueryGraphBuilder(bert_similarity=False)

## Query generator
query_generator = QueryGenerator()

df = pd.read_csv('../data/test-lcquad.csv')

df['answer'] = df['question'].map(ask)


