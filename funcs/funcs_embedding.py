import gensim
import logging



def train_embeddings(txt_file, vec_size=150):

    def read_input(input_file):
        with open (input_file, 'rb') as f:
            for i, line in enumerate (f): 
                if (i%10000==0):
                    logging.info ("read {0} reviews".format (i))
                # do some pre-processing and return a list of words
                yield gensim.utils.simple_preprocess(line)
        logging.info("Done reading data file")

    documents = list(read_input(txt_file))
        
    model = gensim.models.Word2Vec(documents, vector_size=vec_size, window=10, min_count=2, workers=10, sg=1)

    model.wv.save_word2vec_format('out_embedding.bin', binary=True)



def load_embedding():
    wv = gensim.models.KeyedVectors.load_word2vec_format("out_embedding.bin", binary=True)
    return wv


# wv.most_similar(positive=["polite"], topn=5) --> list of close words
# wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1) --> vector arithmetic
# wv.similarity(w1="dirty", w2="smelly")    --> 0.89
# wv.doesnt_match(["cat", "dog", "france"]) --> france


def visualize_embedding(wv, word_list):

    def reduce_dimensions(model, num_dimensions=2, words=[]):

        vectors = [] # positions in vector space
        labels = [] # keep track of words to label our data again later
        word_count = 0
        
        # if no word list is given, assume we want to use the whole data in the model
        if(words == []):
            words = model.index_to_key

        for word in words:
            vectors.append(model[word])
            labels.append(word)

        # convert both lists into numpy vectors for reduction
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)

        # reduce using t-SNE
        tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=2)
        vectors = tsne.fit_transform(vectors)

        return vectors, labels

    def plot_with_matplotlib(x_vals, y_vals, labels, words=[]):

        random.seed(0)
        
        x_vals_new = np.array([])
        y_vals_new = np.array([])
        labels_new = np.array([])
        if(words == []):
            # if no word list is given, assume we want to plot the whole data
            x_vals_new = x_vals
            y_vals_new = y_vals
            labels_new = labels
        else:
            for i in range(len(labels)):
                if(labels[i] in words):
                    x_vals_new = np.append(x_vals_new,x_vals[i])
                    y_vals_new = np.append(y_vals_new,y_vals[i])
                    labels_new = np.append(labels_new,labels[i])
        
        plt.figure(figsize=(12, 12))
        plt.scatter(x_vals_new, y_vals_new)

        # apply labels
        for i in range(len(labels_new)):
            plt.annotate(labels_new[i], (x_vals_new[i], y_vals_new[i]))
        
        plt.show()

    vectors, labels = reduce_dimensions(wv, 2, word_list)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
            
    plot_with_matplotlib(x_vals, y_vals, labels, word_list)





























