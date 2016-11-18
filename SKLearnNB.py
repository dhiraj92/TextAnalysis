# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 06:28:24 2016


"""

# import required libraries
import glob
import codecs
import numpy
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
import pickle

# label the movie review data source as positive or negative
SOURCES = [
    ('MoviePosNeg\\neg\*.txt',  'neg'),
    ('MoviePosNeg\\pos\*.txt',  'pos')
]

TESTSOURCE = [
    ('MoviePosNeg\\test\\neg\*.txt',  'neg'),
    ('MoviePosNeg\\test\\pos\*.txt',  'pos')
]

############################################
########### read the movie review data
def read_files (path):
    files = glob.glob(path)
    for file in files:
        # use Unicode text encoding and ignore any errors 
        with codecs.open(file, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text = text.replace('\n', ' ')
            yield file, text


############################################
########### build data frame for scikitlearn
def build_data_frame(path, classification):
    rows = []
    index = []
    #print path
    for file_name, text in read_files(path):        
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

def Kcross(data,pipeline):
    k_fold = KFold(n=len(data), n_folds=2)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values.astype(str)
    
        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values.astype(str)
    
        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)
    
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='pos')
        scores.append(score)
        
def train():
    ############################################
    ########### main execution
    
    
    testData = DataFrame({'text': [], 'class': []})
    for path, classification in TESTSOURCE:
        testData = testData.append(build_data_frame(path, classification))
    testData = testData.reindex(numpy.random.permutation(testData.index))
    print "Test Data Read"
    
    # read the movie review data 
    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))
    data = data.reindex(numpy.random.permutation(data.index))
    print "Train Data Read"
    
    
    
    # create the processing pipeline
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    pipeline = Pipeline([
        ('vect',    CountVectorizer(ngram_range=(1,2),stop_words='english',lowercase=True)),
        ('tfidf',   TfidfTransformer(smooth_idf=True,use_idf=True)),
        ('clf',     MultinomialNB(alpha=1.0,fit_prior=True))
    ])
    
    
    pipeline.fit(data['text'].values,data['class'].values.astype(str))
    print "Training done"
#    train_indices = 1000
    predict = pipeline.predict(testData['text'].values)
    score = pipeline.score(testData['class'].values.astype(str),predict)
    # print the accuracy
    print('Accuracy score on the test data :', score)
    #saving the pipeline to be used in future
    with open('predictionModel.pkl', 'wb') as f:
        pickle.dump(pipeline, f)



    


def prediction(l=None):
    #to classify our sentence
    print("loading model")
    pipeline = pickle.load(open('predictionModel.pkl'))
    #test sentence if l in none
    Sentence = "This is a beautiful day"
    print("predicting the sentiment" )
    if l is None:
        print("the sentiment is", pipeline.predict([Sentence]).values)
    else:
        print("the sentiment for the list of sentences",pipeline.predict(l).tolist())
        
if __name__ == '__main__':
    #call this function to train the function, comment it out if you just want predict    
    train()
    #pass a list of sentences which we need to classify
    listOfSentences = ['This is a horrible day','This is a beautiful day']    
    prediction(listOfSentences)
    dataset='mnist.pkl.gz'
    #predict(dataset,n_hidden,n_in,n_out)
    #params = test_mlp()
