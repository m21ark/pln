from collections import defaultdict
from nltk import ngrams
import random

def train_ngram(n = 5):

    # Create a placeholder for the model
    n_model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count the frequency of each ngram
    for sentence in reuters.sents():
        for w in range(2, n+1):
            for ngram in ngrams(sentence, n, pad_right=True, pad_left=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
                n_model[ngram[:w-1]][ngram[w-1]] += 1
                
    # Let's transform the counts to probabilities
    for ngram in n_model:
        total_count = float(sum(n_model[ngram].values()))
        for w in n_model[ngram]:
            n_model[ngram][w] = n_model[ngram][w] / total_count

    return n_model

def ngram_guess_next(model, word_list):
    max(n_model[text], key=n_model[text].get)
                
def ngram_complete_sentence(model, text):
    while text[-1] != "</s>":
            
            # select a random probability threshold
            r = random.random()
            
            # select word above the probability threshold, conditioned to the previous word text[-1]
            accumulator = .0
            for word in n_model[(text[-3], text[-2], text[-1])]:
                accumulator += n_model[(text[-3], text[-2], text[-1])][word]
                if accumulator >= r:
                    text.append(word)
                    break
                
    print (' '.join([t for t in text if t]))


