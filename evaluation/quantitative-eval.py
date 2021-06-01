import os
from shutil import Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append(os.path.join(sys.path[0], '..'))
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
    EMBEDDINGS_PATH = '../data/glove.840B.300d.gensim'
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
    print('KGQA - q:', question)
    start = time.time()
    # classify question with pattern
    patterns, raw_probs = pattern_classifier.transform(question)
    print('Pattern:', patterns[0])

    # extract and link entities
    entities, texts = entity_extractor.extract(question)
    print('Extracted entities:', entities)
    try:
        # query graph construction
        Q = query_graph_builder.build(question, entities, texts, patterns[0])

        # build SPARQL query and retrieve answers
        answers_df = query_generator.generate_and_ask(question, Q, entities)
        print(answers_df['Answers'].values)
        return time.time() - start, answers_df['Answers'].values
    except:
        return None, None

def askFTQA(free_text, question, span):
    """
    from span
    """
    print('Free text from span - q:', question)
    start = time.time()
    try:
        answer = free_text.answerFromSpan(question, span)
        print(answer)
        return time.time() - start, answer
    except:
        return None, None

def askFTQAwiki(free_text, question):
    """
    from span
    """
    print('Free text wiki - q:', question)
    start = time.time()
    try:
        answer = free_text.answerFromWiki(question, top=3)
        print(answer)
        return time.time() - start, answer
    except:
        return None, None

def askFTQAnernel(free_text, question, entity_extractor):
    """
    ner nel
    """
    print('Free text ner nel - q:', question)
    if entity_extractor is None:
        print("ERROR: run with action=all for FTQ NER NEL", file=sys.stderr)
        sys.exit(1)
    start = time.time()
    # extract and link entities
    entity, text = entity_extractor.extractMain(question)
    try:
        # get answers from wikipedia
        answer = free_text.answerFromWiki(question, entity=entity, top=3)
        print(answer)
        return time.time() - start, answer
    except:
        return None, None
    

def gen_col(df, index):
    for i in df:
        yield i[index]


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--action', type=click.Choice(['all', 'kgqa', 'ftqa'], case_sensitive=False))
def main(input, output, action):
    pattern_classifier = None
    entity_extractor = None
    query_graph_builder = None
    query_generator = None
    free_text = None

    df = pd.read_csv(input)

    if action in ['kgqa', 'all']:
        pattern_classifier, entity_extractor, query_graph_builder, query_generator = loadKGQA()
        applied_df = df['question'].apply(lambda x: askKGQA(pattern_classifier, entity_extractor, query_graph_builder, query_generator, x))
        df['kg_time'] = list(gen_col(applied_df, 0))
        df['kg_answer_predicted'] = list(gen_col(applied_df, 1))
    if action in ['ftqa', 'all']:
        free_text = loadFTQA()
        # span
        applied_df = df.apply(lambda x: askFTQA(free_text, x['question'], x['span']), axis=1)
        df['ft_time'] = list(gen_col(applied_df, 0))
        df['ft_answer_predicted'] = list(gen_col(applied_df, 1))
        # wiki
        applied_df3 = df.apply(lambda x: askFTQAwiki(free_text, x['question']), axis=1)
        df['ft_wiki_time'] = list(gen_col(applied_df3, 0))
        df['ft_wiki_answer_predicted'] = list(gen_col(applied_df3, 1))
    if action in ['all']:
        # ner nel
        applied_df2 = df.apply(lambda x: askFTQAnernel(free_text, x['question'], entity_extractor), axis=1)
        df['ft_nernel_time'] = list(gen_col(applied_df2, 0))
        df['ft_nernel_answer_predicted'] = list(gen_col(applied_df2, 1))

    df.to_csv(output, index=False)










if __name__ == "__main__":
    main()
