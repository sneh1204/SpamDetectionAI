import random

import joblib
import pandas as pd
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from spacy.util import minibatch

data_path = "data.csv"

sns.set(style="darkgrid")


def train_model(model1, train_data, optimizer, batch_size, epochs=10):
    losses = {}
    random.seed(1)

    for epoch in range(epochs):
        random.shuffle(train_data)

        batches = minibatch(train_data, size=batch_size)
        for batch in batches:
            texts, labels = zip(*batch)

            model1.update(texts, labels, sgd=optimizer, losses=losses)

    return losses['textcat']


def get_predictions(model1, texts):
    docs = [model1.tokenizer(text) for text in texts]

    textcat = model1.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    predicted_labels = scores.argmax(axis=1)
    predicted_class = [textcat.labels[label] for label in predicted_labels]

    return predicted_class


def textChecker(model, text) -> str:
    return 'ham' if model.predict(text) == [0] else 'spam'

def main():

    data = pd.read_csv(data_path)
    observations = len(data.index)

    nlp = spacy.blank("en")

    text_cat = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "bow"})

    nlp.add_pipe(text_cat)

    text_cat.add_label("ham")
    text_cat.add_label("spam")

    x_train, x_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.33, random_state=7)
    train_lables = [{'cats': {'ham': label == 'ham',
                              'spam': label == 'spam'}}  for label in y_train]
    test_lables = [{'cats': {'ham': label == 'ham',
                             'spam': label == 'spam'}} for label in y_test]

    train_data = list(zip(x_train, train_lables))
    test_data = list(zip(x_test, test_lables))

    optimizer = nlp.begin_training()
    batch_size = 5
    epochs = 10

    # joblib.dump(nlp, 'Spam_model.sav')
    # train_model(nlp, train_data, optimizer, batch_size, epochs)

    nlp = joblib.load('Spam_model.sav')
    train_predictions = get_predictions(nlp, x_train)
    test_predictions = get_predictions(nlp, x_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print("Train accuracy: {}".format(train_accuracy))
    print("Test accuracy: {}".format(test_accuracy))

    cf_train_matrix = confusion_matrix(y_train, train_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_train_matrix, annot=True, fmt='d')
    plt.plot()
    plt.show()

    cf_test_matrix = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_test_matrix, annot=True, fmt='d')
    plt.plot()
    plt.show()

    classifiers(data['text'], data['label'])


def classifiers(X, y):
    print("\nTree Learning")
    print("-------------------\n")

    dtclass = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight='balanced')
    rfclass = RandomForestClassifier(warm_start=True, oob_score=True, class_weight='balanced')

    models = [dtclass, rfclass]
    names = ["Decision Tree Classifier", "Random Forest Classifier"]
    vc = TfidfVectorizer()
    # print(X[3])
    X = vc.fit_transform(X)
    y = y.apply(lambda x: 1 if x == 'spam' else 0)
    # print(y.value_counts())

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # print(x_test.shape)
    for name, model in zip(names, models):
        print(name)
        model.fit(x_train, y_train)
        train_score = accuracy_score(y_train, model.predict(x_train))
        test_score = accuracy_score(y_test, model.predict(x_test))
        print("Train Accuracy: {}, Test Accuracy: {}\n".format(train_score, test_score))
        # print(classification_report(y_test, model.predict(x_test)))
    text = input("Enter the text that you want to check :: \n")
    text = vc.transform([text])
    # print(text.shape,"\n")
    print(textChecker(models[1], text))


main()
