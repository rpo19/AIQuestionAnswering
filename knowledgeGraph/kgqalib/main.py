import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
from patternBuilder.SQPBuilder import SQPBuilder
from queryGraphBuilder.QueryGraphBuilder import QueryGraphBuilder
from queryGenerator.QueryGenerator import QueryGenerator
import shared.utils as utils


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

while True:

    # get question from user input
    question = input('Question:')

    # classify question with pattern
    patterns, raw_probs = pattern_classifier.transform(question)
    print('Pattern:', patterns[0])

    # extract and link entities
    entities, texts = entity_extractor.extract(question)
    print('Extracted entities:', entities)

    # query graph construction
    Q = query_graph_builder.build(question, entities, patterns[0])

    # build SPARQL query and retrieve answers
    answers_df = query_generator.generate_and_ask(question, Q)

    # print answers
    utils.print_from_df(answers_df)



