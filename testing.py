import numpy as np
import pandas as pd
from word2vec import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


"""
processing to create training.csv which have:
    index, word1, word2, relation, 
    vector1(word vectorization from word2vec), 
    vector2(word vectorization from word2vec)
"""
# testing set
noun_set = pd.read_table("txt/400_noun_pairs.txt")
verb_set = pd.read_table("txt/400_verb_pairs.txt")
adj_set = pd.read_table("txt/600_adj_pairs.txt")

test_set = pd.concat([noun_set, verb_set, adj_set])


word2vec_dict = dict(zip(word2vec["word"], word2vec["vector"]))

def get_word_vector(word):
    return word2vec_dict.get(word, np.random.rand(300)) 

test_set["Vector1"] = test_set["Word1"].apply(get_word_vector)
test_set["Vector2"] = test_set["Word2"].apply(get_word_vector)

test_set = test_set.dropna(subset=["Vector1", "Vector2"])

rows_with_null = test_set[test_set.isnull().any(axis=1)]
if not rows_with_null.empty:
    print("Rows with null elements:")
    print(rows_with_null)
else:
    print("No rows contain null elements.")

print(test_set)

# store at txt folder
test_set.to_csv("txt/testing_data.csv", index=False)