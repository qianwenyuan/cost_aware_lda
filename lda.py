from sklearn.decomposition import LatentDirichletAllocation
from data import Data

class LDA:
    def __init__(self, n_topic=30, learning_method='batch', max_iter=50):
        self.n_topic = n_topic
        self.learning_method = learning_method
        self.max_iter = max_iter

        self.lda = LatentDirichletAllocation(n_components=n_topic, max_iter=max_iter, learning_method=learning_method)

    def fit(self, tf):
        print('LDA fitting...\n')
        self.lda.fit(tf)

    def print_top_words(self, feature_names=None, n_top_words=20):
        if feature_names:
            for topic_idx, topic in enumerate(self.lda.components_):
                print("Topic #{}".format(topic_idx))
                print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(self.lda.components_)

data = Data()

data.load(sample=True)
data.textPre('w')
tf = data.saveModel('w')

def lda_train(n_topics):
    #print(n_topics)
    model = LDA(int(n_topics))
    model.fit(tf)

    return model.lda.perplexity(tf)

