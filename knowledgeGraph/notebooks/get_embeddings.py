import time
import numpy as np
import pandas as pd

import time
start = time.time()
dims = 200
dtypes = {i:'float32' for i in range(1,dims+1)}
df = pd.read_csv('../../data/glove.twitter.27B.200d.txt', sep=' ', header=None, index_col=0, dtype=dtypes)
stop = time.time()
print("elapsed", stop -start)
df.to_pickle('../../data/glove.twitter.27B.200d.npy')