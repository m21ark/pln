# PLN

## TODO

### Explore --> JONY

- fazer word clouds com base no tf-idf em vez de ser com base em frequencia
- ver se ha correlacao de uso de CAPS com sentimentos de raiva por ex
- ver ratio de palavras negativas/positivas em cada review e ver se ha correlacao com sentimentos --> vader

### PRE-PROCESS

- Apply a basic tokenizer + Do MWE (multi-word expression tokenizer)
- Do name entity recognition and replace them with a single token --> maybe should be done before the cleaning
- experimentar com stemming ou lemma. os 2 juntos em principo da asneira

- Try the spacy processing pipeline as comparison to our own
- Explore alternatives for the negation handling

### MODELOS

- Do POS tagging ?
- Trainar o nosso embedding a 100% dos dados

TESTAR MODELOS COM APPROACH EM PARES E OUTRO COM 1 OUT OF 5 MOST LIKELY
how to combine the word embedding into phrases (max, min, sum, multiply ...)

## Group

- João Alves (up202007614)
- Marco André (up202004891)
- Rúben Monteiro (up202006478)

## Questions

How to deal with negation? Is it really the best way to use not_.
Perguntar se é boa ideia remover palavras com 2< chars.
Do spell check and fix words?
How to train with rep of POS
