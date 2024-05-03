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
# Load data
antonym = pd.read_table("txt/Antonym_vietnamese.txt", header=None)
synonym = pd.read_table("txt/Synonym_vietnamese.txt", header=None)

antonym = antonym[0].str.split(expand=True)
antonym.columns = ["Word1", "Word2"]
antonym["Relation"] = "ANT"

synonym = synonym[0].str.split(expand=True)
synonym.columns = ["Word1", "Word2"]
synonym["Relation"] = "SYN"

train_set = pd.concat([antonym, synonym])

word2vec_dict = dict(zip(word2vec["word"], word2vec["vector"]))

def get_word_vector(word):
    return word2vec_dict.get(word, np.random.rand(150)) 

train_set["Vector1"] = train_set["Word1"].apply(get_word_vector)
train_set["Vector2"] = train_set["Word2"].apply(get_word_vector)

train_set = train_set.dropna(subset=["Vector1", "Vector2"])

# rows_with_null = train_set[train_set.isnull().any(axis=1)]
# if not rows_with_null.empty:
#     print("Rows with null elements:")
#     print(rows_with_null)
# else:
#     print("No rows contain null elements.")

# store at txt folder
train_set.to_csv("txt/training_data.csv", index=False)
