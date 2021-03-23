import time
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate


"""
Preprocess Natural Language question
"""
def preprocess(doc):
    doc = doc.lower().replace('?', ' ?')
    return doc


def load_embeddings(path='../../data/glove.twitter.27B.200d.txt'):
    embeddings_dict = {}
    print('Loading embeddings...')
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def generate_embeddings(path='../../data/glove.twitter.27B.200d.txt'):
    start = time.time()
    dims = 200
    dtypes = {i:'float32' for i in range(1,dims+1)}
    embeddings = load_embeddings(path)
    stop = time.time()
    print("elapsed", stop - start)
    pickle.dump(embeddings, open('../../data/glove.twitter.27B.200d.pickle', 'wb'))

def print_from_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


