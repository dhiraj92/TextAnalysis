# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:39:25 2016

@author: Dhiraj
"""
import nltk
from gensim import corpora
import gensim
from collections import defaultdict
from pprint import pprint
import pdb
def topicExtraction(documents):
    
    stoplist = set(nltk.corpus.stopwords.words("english"))
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
    
    # remove words that appear only once

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    
      # pretty-printer
    #pprint(texts)
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print dictionary
    lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=1,distributed=True)
    #print lda.print_topics(1)
    #model = gensim.models.HdpModel(corpus, id2word=dictionary)
    #print model.print_topics()
    top_words = [[i for i,word in lda.show_topic(topicno,topn=50)] 
                for topicno in range(lda.num_topics)]
    top_words_val = [[word for i,word in lda.show_topic(topicno,topn=50)] 
                for topicno in range(lda.num_topics)]
    #pdb.set_trace()
    #print top_words
    #print top_words_val
    topicsDict = {}    
    for i in range(0,len(top_words)):
        for j in range(0,len(top_words[0])):
            topicsDict[top_words[i][j].encode('utf-8')] = top_words_val[i][j]  
    #print topicsDict
    return topicsDict

if __name__ == "__main__":
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",              
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    topicExtraction(documents)
    