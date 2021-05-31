import os
from shutil import Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from knowledgeGraph.kgqalib.dbpediaNEL.DBPediaEntityExtractor import DBPediaEntityExtractor
from knowledgeGraph.kgqalib.patternBuilder.SQPBuilder import SQPBuilder
from knowledgeGraph.kgqalib.queryGraphBuilder.QueryGraphBuilder import QueryGraphBuilder
from knowledgeGraph.kgqalib.queryGenerator.QueryGenerator import QueryGenerator
import pandas as pd
import click
import time

from text.ftqalib.ftqa.FreeTextQA import FreeTextQA


def loadKGQA():
    ## KGQA 
    MODEL_PATH = '../data/models/pattern_classifier.h5'
    EMBEDDINGS_PATH = './data/glove.840B.300d.gensim'
    GRAPH_BUILDER_MODE = 'glove' # 'glove' | 'stacked' | 'sentence_roberta'
    NER_MODE = 'spotlight' # 'spotlight' | 'custom'

    ## Pattern classifier
    pattern_classifier = SQPBuilder().load(MODEL_PATH)

    ## Entity extractor
    entity_extractor = DBPediaEntityExtractor(mode=NER_MODE)

    ## Query graph builder
    query_graph_builder = QueryGraphBuilder(mode=GRAPH_BUILDER_MODE, path_to_embeddings=EMBEDDINGS_PATH)

    ## Query generator
    query_generator = QueryGenerator()

    return pattern_classifier, entity_extractor, query_graph_builder, query_generator

def loadFTQA():
    ## FTQA
    free_text = FreeTextQA()
    return free_text


def askKGQA(pattern_classifier, entity_extractor, query_graph_builder, query_generator, question):
    start = time.time()
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

    return time.time() - start, answers_df['Answers'].values

def askFTQA(free_text, question, span):
    start = time.time()
    answer = free_text.answerFromSpan(question, span)
    return time.time() - start, answer

def gen_col(df, index):
    for i in df:
        yield i[index]
    

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--action', type=click.Choice(['all', 'kgqa', 'ftqa'], case_sensitive=False))
def main(input, action):
    pattern_classifier = None
    entity_extractor = None
    query_graph_builder = None
    query_generator = None
    free_text = None

    df = pd.read_csv(input)

    if action == 'all':
        pattern_classifier, entity_extractor, query_graph_builder, query_generator = loadKGQA()
        free_text = loadFTQA()
        applied_df = df.apply(lambda x: askFTQA(free_text, x['question'], x['span']))
        df['ft_time'] = list(gen_col(applied_df, 0))
        df['ft_answer_predicted'] = list(gen_col(applied_df, 1))
        applied_df= df['question'].apply(lambda x: askKGQA(pattern_classifier, entity_extractor, query_graph_builder, query_generator, x))
        df['kg_time'] = list(gen_col(applied_df, 0))
        df['kg_answer_predicted'] = list(gen_col(applied_df, 1))
    elif action == 'kgqa':
        pattern_classifier, entity_extractor, query_graph_builder, query_generator = loadKGQA()
        applied_df = df['question'].apply(lambda x: askKGQA(pattern_classifier, entity_extractor, query_graph_builder, query_generator, x))
        df['kg_time'] = list(gen_col(applied_df, 0))
        df['kg_answer_predicted'] = list(gen_col(applied_df, 1))
    else:
        free_text = loadFTQA()
        applied_df = df.apply(lambda x: askFTQA(free_text, x['question'], x['span']))
        df['ft_time'] = list(gen_col(applied_df, 0))
        df['ft_answer_predicted'] = list(gen_col(applied_df, 1))
        
    
    

    





if __name__ == "__main__":
    main()
