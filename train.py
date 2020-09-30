from data import Data
from lda import LDA

data = Data()

data.load()
data.textPre('r')
tf = data.saveModel('r')

model = LDA()
model.fit(tf)
#model.print_top_words(data.tf_vectorizer.get_feature_names())

