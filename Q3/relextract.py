import json
import math

import pandas as pd
import numpy as np
#import spacy

import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

def print_example(data, index):
    """Prints a single example from the dataset. Provided only
    as a way of showing how to access various fields in the
    training and testing data.

    Args:
        data (list(dict)): A list of dictionaries containing the examples 
        index (int): The index of the example to print out.
    """
    # NOTE: You may Delete this function if you wish, it is only provided as 
    #   an example of how to access the data.
    
    # print the sentence (as a list of tokens)
    print("Tokens:")
    print(data[index]["tokens"])

    # print the entities (position in the sentence and type of entity)
    print("Entities:")
    for entity in data[index]["entities"]:
        print("%d %d %s" % (entity["start"], entity["end"], entity["label"]))
    
    # print the relation in the sentence if this is the training data
    if "relation" in data[index]:
        print("Relation:")
        relation = data[index]["relation"]
        print("%d:%s %s %d:%s" % (relation["a_start"], relation["a"],
            relation["relation"], relation["b_start"], relation["b"]))
    else:
        print("Test examples do not have ground truth relations.")

def write_output_file(relations, filename = "q3.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "q3.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)


#def feature_extraction(tokens):


def entity_pairs(entities):
    """
    Returns a list of paired entities
    :param entities: List of entities for a sent
    :return: List of tuples which are pairs of entities
    """
    if not entities:
        return []

    filtered_entities = [entity for entity in entities if entity["label"] == "PERSON" or entity["label"] == "GPE"]
    return list(itertools.combinations(filtered_entities, 2))

def preprocessing(tokens, isalpha, isstop, start, end):
    """
    Performs preprocessing steps in sequence
    * Trimming from start to end
    * Remove digits
    * Remove Stop words
    :param tokens: List of tokens
    :param isalpha: Is each word only alphabetic
    :param isstop: Is each word in the spacy stop word dictionary
    :param start: Start index for trim
    :param end: End index for trim
    :return: List of preprocessed tokens
    """

    if start < 0 :
        start = 0

    tokens_trim = tokens[start : end]
    isalpha_trim = isalpha[start :  end]
    isstop_trim = isstop[start : end]

    proc_sent = []
    for token, alpha, stop in zip(tokens_trim, isalpha_trim, isstop_trim):
        if not stop and alpha :
            proc_sent.append(token)

    if len(proc_sent) == 0:
        #print("Start: %d End: %d"%(start, end))
        print(tokens)

    return proc_sent

def data_extractor(train_data, test = False):
    X = []
    Y = []
    p = []
    for data in train_data:

        # Making a list pairs of entities
        # Storing the start of an entity and end of another entity
        pairs = entity_pairs(data["entities"])
        # print("Length of pairs: " + str(len(pairs)))

        if(len(pairs))> 5 and not test:
            pairs = []
            if (data["relation"]["relation"] == "/people/person/nationality"):

                # Find a start of first and end of the other entity
                start, end = data["relation"]["a_start"], data["relation"]["b_start"]

                if start > end:
                    end, start = start, end
                # Trim the sentences with adding 3 words before start and 3 words after end
                # This might be helpful while finding the context b/w these two entities
                # Preprocessing the data using "isalpha", "isstop" provided with data
                proc_tokens = preprocessing(data["tokens"], data["isalpha"], data["isstop"], start - 3, end + 4)

                # Converting tokens to sentence as CountVectorizer needs a list/iterable of strings
                proc_sent = ' '.join(proc_tokens)

                X.append(proc_sent)
                Y.append(1)

        for pair in pairs:

            if not test:
                # Find if this pair has a relation in ground truth

                if (data["relation"]["relation"] == "/people/person/nationality") and ((pair[0]["start"], pair[1]["start"]) == (data["relation"]["a_start"], data["relation"]["b_start"]) or (pair[1]["start"], pair[0]["start"]) == (data["relation"]["a_start"], data["relation"]["b_start"])):
                    r = 1
                else:
                    r = 0
                Y.append(r)

            # Find a start of first and end of the other entity
            start, end = pair[0]["start"], pair[1]["end"]


            entity1, entity2 = data["tokens"][pair[0]["start"]:pair[0]["end"]], data["tokens"][pair[1]["start"]:pair[1]["end"]]
            entity1, entity2 = ' '.join(entity1).lower(), ' '.join(entity2).lower()
            if(pair[0]["label"] == "GPE"):
                entity2, entity1 = entity1, entity2

            if start > end:
                end, start = start, end


            # Trim the sentences with adding 3 words before start and 3 words after end
            # This might be helpful while finding the context b/w these two entities
            # Preprocessing the data using "isalpha", "isstop" provided with data
            proc_tokens = preprocessing(data["tokens"], data["isalpha"], data["isstop"], start - 3, end + 4)

            # Converting tokens to sentence as CountVectorizer needs a list/iterable of strings
            proc_sent = ' '.join(proc_tokens)

            X.append(proc_sent)
            p.append((entity1, entity2))


    return X, Y, p

# print a single training example
print("Training example:")
print_example(train_data, 1)

print("---------------")
print("Testing example:")
# print a single testing example
# the testing example does not have a ground
# truth relation
print_example(test_data, 2)


X, Y, _ = data_extractor(train_data)

# Initializing BoW model and ngram with min 3 and max 3
vectorizer = CountVectorizer(analyzer = "word", binary = True, ngram_range=(1,3), max_features= 5000)
X_features = vectorizer.fit_transform(X)

# Print top 5 features
print(X_features.shape)
# Distribution of pairs
(unique, counts) = np.unique(Y, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

Y = np.array(Y).reshape(len(Y), 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y.ravel(), test_size=0.1, stratify=Y)


print("Trianing Data: " + str(X_train.shape))
print("Testing Data: " + str(X_test.shape))

penalty = ['l2']
K = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
iterations = [100, 250, 500, 1000, 2000]
c_values = [math.pow(3, k) for k in K]
# define grid search
grid = dict(penalty=penalty,C=c_values, max_iter = iterations)

clf = LogisticRegression()
gridSearch = GridSearchCV(clf, grid, cv=3)
result = gridSearch.fit(X_train, Y_train)
Y_pred = gridSearch.predict(X_test)
# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))
print("Val Accuracy: %f" % f1_score(Y_test, Y_pred))


# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file

# Convert the test data
X, _, pairs = data_extractor(test_data, True)

# Vectorize the data
X_features = vectorizer.transform(X)

# Predict for the testing data
Y_pred = gridSearch.predict(X_features)

# Read the country names from the country_names.txt given
# Using this we can filter the pairs that doesnt have countries in their GPE position.
countries = []
with open('country_names.txt') as f:
    for country in f:
        countries.append(str(country).replace("\n", "").lower())

# Filtering the predictions with Nationality
relations = [pairs for pred, pairs in zip(Y_pred, pairs) if pred == 1]
relations = [r for r in relations if r[1] in countries and r[0] not in countries]
"""relations = [
    ('Hokusai', 'Japan'), 
    ('Hans Christian Andersen', 'Denmark')
    ]
"""
write_output_file(relations)
