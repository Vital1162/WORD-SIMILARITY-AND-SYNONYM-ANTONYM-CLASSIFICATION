import numpy as np
import pandas as pd
from word2vec import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


"""
After we proecssing done the data which in the training.py 
and the testing.py
The result that we now have training.csv and testing.csv
""" 

train_dataset = pd.read_csv("txt/training_data.csv")

test_dataset = pd.read_csv("txt/testing_data.csv")


# fix ANT to 0 and SYN to 1
train_dataset['Relation'] = train_dataset['Relation'].replace({'ANT': 0, 'SYN': 1})
test_dataset['Relation'] = test_dataset['Relation'].replace({'ANT': 0, 'SYN': 1})


print(train_dataset)
print(test_dataset)

# well we change it into np array to process
# example [1 2 3] -> [1,2,3]
def parse_vector(vector_string):
    return np.array([float(val) for val in vector_string.strip('[]').split()])

train_dataset['Vector1'] = train_dataset['Vector1'].apply(parse_vector)
train_dataset['Vector2'] = train_dataset['Vector2'].apply(parse_vector)
test_dataset['Vector1'] = test_dataset['Vector1'].apply(parse_vector)
test_dataset['Vector2'] = test_dataset['Vector2'].apply(parse_vector)

print(train_dataset)
print(test_dataset)

X_train = np.concatenate([train_dataset['Vector1'].values.tolist(),\
    train_dataset['Vector2'].values.tolist()], axis=1)
y_train = train_dataset['Relation'].values

X_test = np.concatenate([test_dataset['Vector1'].values.tolist(),\
    test_dataset['Vector2'].values.tolist()], axis=1)

y_test = test_dataset['Relation'].values

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



