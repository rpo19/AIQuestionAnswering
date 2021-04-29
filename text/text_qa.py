# %% imports
import pandas as pd
import importlib.util 
spec = importlib.util.spec_from_file_location('dbpediaNEL.DBPediaEntityExtractor', '../knowledgeGraph/kgqalib/dbpediaNEL/DBPediaEntityExtractor.py')
DBPediaEntityExtractor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = DBPediaEntityExtractor 
spec.loader.exec_module(DBPediaEntityExtractor)
import ftqalib.ftqa.FreeTextQA as ftqa
# %% setup
# set seed
SEED = 42
# load ftqa model
qa = ftqa.FreeTextQA()
# load dbpedia entity extractor
dbpee = DBPediaEntityExtractor.DBPediaEntityExtractor()
# %% extract wiki answers of a SQuAD sample
# DO NOT RUN WITH TOO MUCH SAMPLES!!! COMPUTATIONAL SUICIDE!!!
N_SAMPLES=8
flatten_squad = pd.read_csv('../data/flatten_squad.csv')
# TODO: not random sample, should be more appropriate selecting only P1?
flatten_squad = flatten_squad.sample(N_SAMPLES, random_state=SEED)
flatten_squad['wikiAnswerInfo'] = flatten_squad['question']\
        .apply(lambda x: qa.answerFromWiki(x, dbpee.extractMain(x)[0], True))
#flattend_squad.to_csv('../data/flatten_squad_wikianswers.csv', index=False)
# %% clean wiki answers
#flatten_squad = pd.read_csv('../data/flatten_squad_wikianswers.csv')
#  split answer info clearly
flatten_squad['wikiResource'] = flatten_squad['wikiAnswerInfo']\
        .apply(lambda x: x['entity'] if x is not None else x)
flatten_squad['wikiAnswer'] = flatten_squad['wikiAnswerInfo']\
        .apply(lambda x: x['answer'] if x is not None else x)
# uniform entities
flatten_squad['resource'] = flatten_squad['resource']\
        .apply(lambda x: x.replace('_', ' '))
# reorder columns for readability
flatten_squad[
        ['resource',
        'wikiResource',
        'question',
        'answers',
        'wikiAnswer',
        'wikiAnswerInfo']]
# %% compare wiki answers
flatten_squad[flatten_squad['answers'] == flatten_squad['wikiAnswer']]
# %% compare wiki entities
flatten_squad[flatten_squad['resource'] == flatten_squad['wikiResource']]


# %% extract wiki answers of a LC-QAD sample
lc_quad = pd.read_json('../data/test-data-answers.json')
# DO NOT RUN WITH TOO MUCH SAMPLES!!! COMPUTATIONAL SUICIDE!!!
N_SAMPLES=8
# TODO: not random sample, should be more appropriate selecting only P1?
lc_quad = lc_quad.sample(N_SAMPLES, random_state=SEED)
lc_quad['wikiAnswerInfo'] = lc_quad['question']\
        .apply(lambda x: qa.answerFromWiki(x, dbpee.extractMain(x)[0], True))
#lc_quad.to_csv('../data/lc_quad_wikianswers.csv', index=False)

# %% clean wiki answers
#lc_quad = pd.read_csv('../data/lc_quad_wikianswers.csv')
#  split answer info clearly
lc_quad['wikiAnswer'] = lc_quad['wikiAnswerInfo']\
        .apply(lambda x: x['answer'] if x is not None else x)
lc_quad['wikiResource'] = lc_quad['wikiAnswerInfo']\
        .apply(lambda x: x['entity'] if x is not None else x)
lc_quad
# %% compare wiki answers
lc_quad[lc_quad['answer'] == lc_quad['wikiAnswer']]

# %%
