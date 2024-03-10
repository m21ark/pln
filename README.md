# PLN

## TODO

### Explore

- fazer word clouds com base no tf-idf em vez de ser com base em frequencia
- ver se ha correlacao entre palavras de negacao com sentimentos negativos
- ver se ha correlacao de uso de CAPS com sentimentos de raiva por ex
- ver ratio de palavras negativas/positivas em cada review e ver se ha correlacao com sentimentos
- ver ration de palavras com conatacao positiva e negativa no dataset e ver se ha alguma trend interessante

### PRE-PROCESS

- Do spell check and fix words
- Do MWE (multi-word expression tokenizer) detection and replace them with a single token
- Do name entity recognition and replace them with a single token --> maybe should be done before the cleaning
- Do POS tagging ?
- Lemmatize + Stem is giving some strange results for some words
- Try the spacy processing pipeline
- Explore alternatives for the negation handling

### Representations

    text --> tmb supostamente vai ser pessimo

    custom--> podemos tentar fazer a nossa propria rep com base na exploracao

    BOW model --> use countvectorizer
    1-HOT --> mto lixo mas pronto
    tf-idf

    NGRAMS:
        usar diferentes tamanhos
    
    EMBEDDINGS:

    usar embeddings ja feitos (word2vec)
    usar nossos proprios embeddings --> fazer testes para ver se é bom com similaridade, doesnt match, queen/capital check ...
    --> experimentar com diferentes tamanhos de embeddings

    usar append, max embedding (max should be better) aggregators

REGULARIZATION --> technique para evitar overfitting by penalizing excessive feature weights:

    Logistical regression supports L1,L2 (default) regularizations as hyperparamters:
            clf = LogisticRegression(penalty='l2')  
            
            L1 dá mais sparse, podemos usar mix de L1, L2 em diferentes qntd
            
            clf_l1 = LogisticRegression(penalty='l1', solver='saga')    
            clf_mix = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)

            LinearSVC(penalty='l2') 

            --> verificar numero de pesos não nulos 
                num_nonzero_weights = np.count_nonzero(clf.coef_)
                print("Number of non-zero weights:", num_nonzero_weights)

TESTAR MODELOS COM APPROACH EM PARES E OUTRO COM 1 OUT OF 5 MOST LIKELY

## Group

- João Alves (up20200XXXX)
- Marco André (up202004891)
- Rúben Monteiro (up20200XXXX)
