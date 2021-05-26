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

def load_models_kgqa(load_pattern_classifier = True, load_entity_extractor = True):
    MODEL_PATH = '../../data/models/pattern_classifier.h5'
    EMBEDDINGS_PATH = '../../data/glove.840B.300d.gensim'
    GRAPH_BUILDER_MODE = 'glove' # 'glove' | 'stacked' | 'sentence_roberta'
    NER_MODE = 'spotlight' # 'spotlight' | 'custom'

    pattern_classifier = None
    entity_extractor = None

    # instantiate modules
    ## Pattern classifier
    if load_pattern_classifier:
        pattern_classifier = SQPBuilder().load(MODEL_PATH)

    ## Entity extractor
    if load_entity_extractor:
        entity_extractor = DBPediaEntityExtractor(mode=NER_MODE)

    ## Query graph builder
    query_graph_builder = QueryGraphBuilder(mode=GRAPH_BUILDER_MODE, path_to_embeddings=EMBEDDINGS_PATH)

    ## Query generator
    query_generator = QueryGenerator()

    return pattern_classifier, entity_extractor, query_graph_builder, query_generator

def ask_kgqa(question, pattern_classifier, entity_extractor, query_graph_builder, query_generator, true_entities = None):
    # TODO ignore exceptions
    # classify question with pattern
    patterns, _ = pattern_classifier.transform(question)
    print('Done pattern')
    
    # extract and link entities
    if entity_extractor:
        entities, texts = entity_extractor.extract(question)
    elif true_entities:
        entities = true_entities
        texts = None
    else:
        raise ValueError('No entities provided')
    
    
    if not entities:
        print("Could not identify any entity.")
        return None, patterns[0], None

    entities_copy = entities.copy()

    # query graph construction
    Q = query_graph_builder.build(question, entities_copy, texts, patterns[0])



    if not Q:
        print("Could not create a query graph.")
        return None, patterns[0], entities

    SPARQL_query, _ = query_generator.generate(question, Q)



    SPARQL_query.replace('\n', ' ')

    return SPARQL_query, patterns[0], entities

def ask_all(df, output, pattern_classifier, entity_extractor, query_graph_builder, query_generator):
    length = df.shape[0]

    print("Start asking - ALL PHASES...")
    COUNT=0
    with open(output, 'a') as fw:
        for i, row in df.iterrows():
            md = row.to_dict()
            md['pd_index'] = i
            try:
                print(f'\r{COUNT}/{length}', end='')
                print()
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

def ask_from_entities(df, output, pattern_classifier, entity_extractor, query_graph_builder, query_generator):
    length = df.shape[0]

    print("Start asking - FROM ENTITIES...")
    COUNT=0
    with open(output, 'a') as fw:
        for i, row in df.iterrows():
            md = row.to_dict()
            md['pd_index'] = i
            try:
                print(f'\r{COUNT}/{length}', end='')
                COUNT += 1
                question = row["corrected_question"]
                true_entities = ['<' + entity + '>' for entity in row['correct_entities']]

                query, pattern, entities = ask_kgqa(question, pattern_classifier, entity_extractor, 
                    query_graph_builder, query_generator, true_entities=true_entities)

                md['predicted_query'] = query
                md['predicted_pattern'] = pattern
                md['predicted_entities'] = entities
            except Exception as e:
                print('Exception ', e, 'on index', i)
                md['error'] = 1
            fw.write(json.dumps(md)+'\n')


        print("\nDone :)")

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--skip', type=click.Choice(['none', 'extract-entities'], case_sensitive=False), default='none',
    help='From which phase you want to start predicting')
@click.option('--start-from', default=0, help='Start evaluation from line number.')
def main(input, output, skip, start_from):
    """
    Given as input the csv file obtained from `sqppatterns prepare-evaluation`,
    it creates the sparql queries starting from the natural language question
    saving data in the output file in json format
    """
    _, file_extension = os.path.splitext(input)
    print("Loading input csv")
    if file_extension == '.csv':
        df = pd.read_csv(input, index_col=0)
    elif file_extension == '.jsonl':
        df = pd.read_json(input, lines=True)
        df = df.iloc[int(start_from):]
    else:
        raise ValueError('Extension supported are csv and jsonl.')

    load_entity_extractor = False if skip == 'extract-entities' else True

    print("Loading KGQA")
    pattern_classifier, entity_extractor, query_graph_builder, query_generator = load_models_kgqa(load_entity_extractor=load_entity_extractor)
    print("Loaded KGQA :)")

    if load_entity_extractor:
        ask_all(df, output, pattern_classifier, entity_extractor, query_graph_builder, query_generator)
    else:
        ask_from_entities(df, output, pattern_classifier, entity_extractor, query_graph_builder, query_generator)

    

if __name__ == "__main__":
    main()