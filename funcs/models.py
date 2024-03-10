from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt

# Models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


# Model Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ====================== MODELS ======================

def model_bow(corpus, max_features = 1500):
    vectorizer = CountVectorizer(max_features = max_features)
    x = vectorizer.fit_transform(corpus).toarray()
    return x

def model_one_hot(corpus):
    vectorizer_binary = CountVectorizer(binary=True)
    x = vectorizer_binary.fit_transform(corpus).toarray()    
    return x

def model_tf_idf(corpus):
    vectorizer_tfidf = TfidfVectorizer()
    x = vectorizer_tfidf.fit_transform(corpus).toarray()
    return x

def model_ngram(corpus, ngram_range = (1,2)):
    vectorizer_bigram = CountVectorizer(ngram_range = ngram_range)
    x = vectorizer_bigram.fit_transform(corpus).toarray()
    return x

models = [
    lambda: MultinomialNB(), # Naive Bayes
    lambda: DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42),
    lambda: GaussianNB(),
    lambda: KNeighborsClassifier(n_neighbors=15, weights='uniform'),
    lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42),
    lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42), # Neural Network
    lambda: SVC(C=0.1, kernel='linear', probability=True), # Support Vector Machine
    lambda: LogisticRegression(penalty='l2'), # Linear Model with overfitting avoidance
    lambda: LinearSVC(penalty='l2'), # Linear Model with overfitting avoidance

    # Linear Model with stochastic gradient descent learning (loss function)
    lambda: SGDClassifier(loss='log', penalty='l2', alpha=0.001, max_iter=100, random_state=42)
]

# ====================== SENTIMENT ANAYZER ======================

def model_vader(corpus):
    # VADER is a robust rule-based lexicon tool tuned to assess social media sentiment 
    # Returns a binary result for each phrase in the corpus where 1 is positive
    analyzer = SentimentIntensityAnalyzer()
    x = []
    for rev in corpus:
        x.append(1 if analyzer.polarity_scores(rev)['compound'] > 0 else 0)
    return x


# ====================== TRAIN & EVALUATE ======================

def split(x, y, test_size = 0.2, log=False):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 0, stratify=y)

    if log:
        print("Train Shape:")
        print(X_train.shape, y_train.shape)
        print("Test Shape:")
        print(X_test.shape, y_test.shape)

        print("\nLabel distribution in the training set:")
        print(y_train.value_counts())
        print("\nLabel distribution in the test set:")
        print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def evaluate(y_test, y_pred):
    # confusion matrix
    print(confusion_matrix(y_test, y_pred))

    # accuracy, precision, recall, f1
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))


def train(model, x, y, split_size = 0.2, cross_count = 0):
    if cross_count == 0:
        X_train, X_test, y_train, y_test = split(x, y, test_size = split_size)

        # Train and evaluate model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate(y_test, y_pred)
    else:
        scores = cross_validate(model, x, y, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], return_train_score=True)
        print(scores)

def show_cm(cm):
    # cm = np.array([[TP, FP], [FN, TN]])
    classes = ['Playoff', 'Eliminated']
    plt.matshow(cm)
    plt.suptitle('Confusion matrix')
    total = sum(sum(cm))
    plt.title('Total cases: {}'.format(total))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            perc = round(cm[i, j] / total * 100, 1)
            plt.text(j, i, f"{format(cm[i, j], '.0f')} : {perc}%", horizontalalignment="center",
                     color="black" if cm[i, j] > cm.max() / 2 else "white")

    plt.show()









