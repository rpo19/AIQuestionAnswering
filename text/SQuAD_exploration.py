# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% load SQuAD data
squad_test = pd.read_json('../data/dev-v1.1.json')
squad_train = pd.read_json('../data/train-v1.1.json')

# %% SQuAD data analysis: TRAIN
# number of questions
count = 0
length = 0
max_length = 0
min_length = 10000
lengths = []
for key, value in squad_train.iterrows():
        for paragraph in value['data']['paragraphs']:
                for qas in paragraph['qas']:
                        count += 1
                tmp = len(paragraph['context'].split())
                lengths.append(tmp)
print('# entries: ',len(squad_train))      
print('# questions: ', count)
print('# avg span length: ', np.mean(lengths))
print('# max span length: ', max(lengths))
print('# min span length: ', min(lengths))
plt.hist(lengths, bins=100)
plt.show()
# %% SQuAD data analysis: TEST
# number of questions
count = 0
length = 0
max_length = 0
min_length = 10000
lengths = []
for key, value in squad_test.iterrows():
        for paragraph in value['data']['paragraphs']:
                for qas in paragraph['qas']:
                        count += 1
                tmp = len(paragraph['context'].split())
                lengths.append(tmp)
print('# entries: ',len(squad_test))      
print('# questions: ', count)
print('# avg span length: ', np.mean(lengths))
print('# max span length: ', max(lengths))
print('# min span length: ', min(lengths))
plt.hist(lengths, bins=100)
plt.show()
# %% flatten data TEST
labels = ['resource', 'question', 'answers' ]
flatten_squad = pd.DataFrame(columns=labels)
for key, value in squad_test.iterrows():
        for paragraph in value['data']['paragraphs']:
                for qas in paragraph['qas']:
                        q = {
                                'resource': value['data']['title'],
                                'question': qas['question'],
                                'answers': qas['answers'][0]['text']
                        }
                        flatten_squad = flatten_squad.append(q, ignore_index=True)
flatten_squad.to_csv('../data/flatten_squad.csv', index=False)


# %%
