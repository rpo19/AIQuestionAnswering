#!/usr/bin/env python
import click
import pandas as pd
import json

import os
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from knowledgeGraph.kgqalib.dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
from knowledgeGraph.kgqalib.patternBuilder.SQPBuilder import SQPBuilder
from knowledgeGraph.kgqalib.queryGraphBuilder.QueryGraphBuilder import QueryGraphBuilder
from knowledgeGraph.kgqalib.queryGenerator.QueryGenerator import QueryGenerator

def load_models_kgqa():
    MODEL_PATH = '../../data/models/pattern_classifier.h5'
    EMBEDDINGS_PATH = '../../data/glove.840B.300d.gensim'
    GRAPH_BUILDER_MODE = 'glove' # 'glove' | 'stacked' | 'sentence_roberta'
    NER_MODE = 'spotlight' # 'spotlight' | 'custom'


    # instantiate modules
    ## Pattern classifier
    pattern_classifier = SQPBuilder().load(MODEL_PATH)

    ## Entity extractor
    entity_extractor = DBPediaEntityExtractor(mode=NER_MODE)

    ## Query graph builder
    query_graph_builder = QueryGraphBuilder(mode=GRAPH_BUILDER_MODE, path_to_embeddings=EMBEDDINGS_PATH)

    ## Query generator
    query_generator = QueryGenerator()

    return pattern_classifier, entity_extractor, query_graph_builder, query_generator

def ask_kgqa(question, pattern_classifier, entity_extractor, query_graph_builder, query_generator):
    # TODO ignore exceptions
    # classify question with pattern
    patterns, _ = pattern_classifier.transform(question)

    # extract and link entities
    entities, texts = entity_extractor.extract(question)
    if not entities:
        print("Could not identify any entity.")
        return None, patterns[0], None

    entities_copy = entities.copy()

    # query graph construction
    Q = query_graph_builder.build(question, entities, texts, patterns[0])

    if not Q:
        print("Could not create a query graph.")
        return None, patterns[0], entities_copy

    SPARQL_query, _ = query_generator.generate(question, Q)

    SPARQL_query.replace('\n', ' ')

    return SPARQL_query, patterns[0], entities_copy

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
def main(input, output):
    print("Loading input csv")
    df = pd.read_csv(input, index_col=0)
    print("Loading KGQA")
    pattern_classifier, entity_extractor, query_graph_builder, query_generator = load_models_kgqa()
    print("Loaded KGQA :)")

    length = df.shape[0]

    print("Start asking...")
    COUNT=0
    with open(output, 'a') as fw:
        for i, row in df.iterrows():
            md = row.to_dict()
            md['pd_index'] = i
            try:
                print(f'\r{COUNT}/{length}', end='')
                COUNT += 1
                question = row["corrected_question"]
                query, pattern, entities = ask_kgqa(question, pattern_classifier, entity_extractor, query_graph_builder, query_generator)

                md['predicted_query'] = query
                md['predicted_pattern'] = pattern
                md['predicted_entities'] = entities
            except Exception as e:
                print('Exception ', e, 'on index', i)
                md['error'] = 1

            fw.write(json.dumps(md)+'\n')


        print("\nDone :)")

if __name__ == "__main__":
    main()