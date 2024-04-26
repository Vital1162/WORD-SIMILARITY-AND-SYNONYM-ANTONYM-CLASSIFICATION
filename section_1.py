import numpy as np
import pandas as pd
from word2vec import word2vec
from sklearn.metrics.pairwise import cosine_similarity


Visim = pd.read_table("txt/Visim-400.txt")

"""
after import the word2vec we
"""

def Cosine(word2vec):
    # browse each row of Visim table
    for _ in range(len(Visim)):
        word_1 = Visim.iloc[_ ,0]
        word_2 = Visim.iloc[_ ,1]
        # can not caculate cosin if they are not include in the w2v file
        if word_1 not in word2vec["word"].values or \
            word_2 not in word2vec["word"].values:
            continue
        
        # vector of word_1 and word_2
        vec_1 = word2vec[word2vec["word"] == word_1]["vector"].values[0]
        vec_2 = word2vec[word2vec["word"] == word_2]["vector"].values[0]

        # reshape to put into cosine_similarity
        vec_1 = vec_1.reshape(1, -1)
        vec_2 = vec_2.reshape(1, -1)

        similarity_matrix = cosine_similarity(vec_1, vec_2)
        print(f"similarity between {word_1} and {word_2}: {similarity_matrix[0][0]}" )

Cosine(word2vec=word2vec)