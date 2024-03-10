import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy

def corpus_pre_process(dataset, target_column):
    corpus = []
    ps = PorterStemmer()
    sw = set(stopwords.words('english'))
    for i in range(0, dataset[target_column].size):
        # Remove non alpha chars 
        review = re.sub('[^a-zA-Z]', ' ', dataset[target_column][i])

        # Convert to lower-case
        review = review.lower()

        # split into tokens, apply stemming and remove stop words
        review = ' '.join([ps.stem(w) for w in review.split() if w not in sw])

        corpus.append(review)

    return corpus

def spacy_pre_process(dataset, target_column):
    nlp = spacy.load("en_core_web_sm")
    corpus = []

    for sentence in range(0, dataset[target_column].size):
        corpus.append(nlp(sentence))

    return corpus

    
