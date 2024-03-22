# PLN - Group J

## TODO

### PRE-PROCESS

- Apply a basic tokenizer + Do MWE (multi-word expression tokenizer)
- Do name entity recognition and replace them with a single token --> maybe should be done before the cleaning
- experimentar com stemming ou lemma. os 2 juntos em principo da asneira

- Try the spacy processing pipeline as comparison to our own
- Explore alternatives for the negation handling

### MODELOS

- Do POS tagging --> fica para dps se nos aptecer
- Treainar o nosso embedding a 100% dos dados

TESTAR MODELOS COM APPROACH EM PARES E OUTRO COM 1 OUT OF 5 MOST LIKELY
how to combine the word embedding into phrases (max, min, sum, multiply ...)

## Group

- João Alves (up202007614)
- Marco André (up202004891)
- Rúben Monteiro (up202006478)

## Questions

How to deal with negation? Is it really the best way to use not_. --> talvez nem abordar a questao de negacao
remover palavras com 1 char.
How to train with rep of POS --> temos de definir as nossas proprias features e ter cuidado com a quantidade de tags q podem dar mto sparse
maybe procurar embeddings pre treinados em tweets:


FastText: FastText is another popular library for word embeddings, developed by Facebook. It's known for its ability to handle out-of-vocabulary words by breaking them down into character n-grams. You can train FastText embeddings on a Twitter corpus to get tweet-specific embeddings.

BERT: BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language representation model developed by Google. While BERT is primarily used for contextual embeddings at the sentence or document level, you can fine-tune pre-trained BERT models on Twitter data to get tweet embeddings.

Tweet2Vec: Tweet2Vec is an embedding method specifically designed for tweets. It's based on paragraph vectors (Doc2Vec) but adapted for Twitter data. It captures the semantics of tweets by considering both the tweet text and the user who posted it.