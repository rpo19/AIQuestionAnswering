from dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
import knowledgeGraph.kgqalib.utils as utils


MODEL_PATH = './data/models/pattern_classifier.h5'

# instantiate modules
## Pattern classifier
pattern_classifier = utils.SQPBuilder().load(MODEL_PATH)

## Entity extractor
entity_extractor = DBPediaEntityExtractor()

## Query graph builder

## Query builder


while True:
    # get question from user input
    question = input('Question:')

    # classify question with pattern
    patterns, raw_probs = pattern_classifier.transform(question)

    # extract and link entities
    entities = entity_extractor.extract(question)

    print(entities)

    # query graph construction

    # query construction

    # retrieve answers



