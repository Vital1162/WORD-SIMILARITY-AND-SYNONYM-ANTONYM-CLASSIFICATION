import pandas as pd
from word2vec import word2vec
from sklearn.metrics.pairwise import cosine_similarity

"""
input: will be the word2vec table, target word,
k is number word with closest cosin length (default is 5)
with condition that the target word must be in word2vec

output: list of word have closest cosin length
""" 
def find_k_nearest_words(word2vec, target_word, k=5):
    #check target word in the word2vcec
    if target_word not in word2vec["word"].values:
        print(f"{target_word} not found in word2vec")
        return []

    # target word vector
    target_vec = word2vec[word2vec["word"] == target_word]["vector"].values[0]

    similarities = []
    for _, row in word2vec.iterrows():
        
        if row["word"] != target_word:
            #compute the consine each word in word2vec with target word
            similarity = cosine_similarity([target_vec], [row["vector"]])[0][0]
            similarities.append((row["word"], similarity))
    #sort the list again 
    similarities.sort(key=lambda x: x[1], reverse=True)
    #only return the range by k
    return [word for word, _ in similarities[:k]]


#some example test
target_word = "táo"
k_nearest_words = find_k_nearest_words(word2vec, target_word, k=6)
print(f"The {len(k_nearest_words)} nearest words to '{target_word}' are:")
print(k_nearest_words)