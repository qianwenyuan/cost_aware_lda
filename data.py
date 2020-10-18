import sys
import os
import string
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
    
def textPrecessing(text):
    #小写化
    text = text.lower()
    #去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, ' ')
    #分词
    wordLst = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    return " ".join(filtered)

class Data:
    def __init__(self, n_samples=5000, textPre_FilePath='./text.txt', tf_ModelPath='./model'):
        self.n_samples = n_samples
        self.textPre_FilePath = textPre_FilePath
        self.tf_ModelPath = tf_ModelPath
    
    def load(self, dataset=None, sample=False):
        print('loading data...\n')

        nsamples = self.n_samples
        if not sample:
            nsamples = -1
        
        if dataset:
            self.dataset = dataset[:nsamples]
        else:
            dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
            self.dataset = dataset.data[:nsamples]

        return self.dataset

    def textPre(self, mode='r'):
        if mode=='r':
            print('reading text from file and processing...\n')
        else:
            print('text processing and writing into file...\n')

        path = self.textPre_FilePath
        docLst = []
        if mode == 'w':
            for desc in self.dataset :
                docLst.append(textPrecessing(desc).encode('utf-8'))
            with open(path, 'w') as f:
                for line in docLst:
                    f.write('{}\n'.format(line))
        else:
            with open(path, 'r') as f:
                for line in f.readlines():
                    if line != '':
                        docLst.append(line.strip())
        self.docLst = docLst

        return docLst

    def saveModel(self, mode='r'):
        print('vectorizing...\n')

        path = self.tf_ModelPath

        if mode=='w':
            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tf = tf_vectorizer.fit_transform(self.docLst)
            joblib.dump(tf_vectorizer, path)
        else:
            tf_vectorizer = joblib.load(path)
            tf = tf_vectorizer.fit_transform(self.docLst)

        self.tf_vectorizer = tf_vectorizer
        self.tf = tf

        return tf

