# %% imports
import pandas as pd 

# %% load data
lc_quad = pd.read_json('../../data/whole.jsonl', orient='records', lines=True)

# %% question types map
question_types = {
    'aggregation': [105, 406, 111, 405, 401, 102, 403, 101, 106, 103, 108, 402],
    'answer-type': [152, 151],
    'select-type': [2, 16, 305, 308, 301, 5, 3, 15, 306, 1, 6, 307, 303, 7, 11, 311, 8, 315, 601]
}

def getQuestionType(sparql_template_id):
    if sparql_template_id in question_types['aggregation']:
        return 'aggregation'
    if sparql_template_id in question_types['answer-type']:
        return 'answer-type'
    if sparql_template_id in question_types['select-type']:
        return 'select-type'
    return 'unexpected sparql_template_id'

# %% test
lc_quad['correct_question_type'] = lc_quad['sparql_template_id'].apply(getQuestionType)
# %% output
print(lc_quad['correct_question_type'].value_counts())

# %% save
lc_quad.to_json('../../data/whole_cbe.jsonl', orient='records', lines=True)


# %%
