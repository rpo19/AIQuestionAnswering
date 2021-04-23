import time
import numpy as np
import pandas as pd
import pickle

def load_embeddings(path='../../data/glove.twitter.27B.200d.txt'):
    embeddings_dict = {}
    print('Loading embeddings...')
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            #word = values[0]
            #vector = np.asarray(values[1:], "float32")
            word = ''.join(values[:-300])
            vector = np.asarray(values[-300:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

start = time.time()
dims = 200
#dtypes = {i:'float32' for i in range(1,dims+1)}
embeddings = load_embeddings('../../data/glove.840B.300d.txt')
stop = time.time()
print("elapsed", stop -start)
pickle.dump(embeddings, open('../../data/glove.840B.300d.pickle', 'wb'))