# PLN - Group J

## TODO

### PRE-PROCESS

- Apply a basic tokenizer + Do MWE (multi-word expression tokenizer)
- Do name entity recognition and replace them with a single token --> maybe should be done before the cleaning
- experimentar com stemming ou lemma. os 2 juntos em principo da asneira

- Try the spacy processing pipeline as comparison to our own
- Explore alternatives for the negation handling

### MODELOS

- Do POS tagging ?
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
maybe procurar embeddings pre treinados em tweets