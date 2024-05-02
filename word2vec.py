import pandas as pd
import numpy as np


"""
processing the word2vec txt
"""
# Read from txt w2c_150
word2vec = pd.read_table("txt/W2V_150.txt", header=None, skiprows=2)

# print(word2vec)
# word2vec[0] = word2vec[0].str.replace('[^\w\s]','')

#split with whitepace
word_vectors = word2vec[0].str.split(expand=True)

# print(word_vectors)

#create colunm word
word2vec["word"] = word_vectors[0]


# create vector by addding alll colunm from 1 to 150 create a numpy array
word2vec["vector"] = word_vectors[word_vectors.columns[1:]]\
    .apply(lambda x: np.array(x.astype(np.float32)), axis=1)


#here we have a word2vec
word2vec = word2vec[["word", "vector"]]
