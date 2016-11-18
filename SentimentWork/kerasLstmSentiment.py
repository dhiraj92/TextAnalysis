# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:57:49 2016

@author: Dhiraj
"""
''' This pipeline demonstrates the use of Word2vec vectors for text
    classification with LSTM (RNN) neural networks. Possibly inclusion of a CNN
    layer on top of this will be useful.
    
    Word2vec (ndim=300) vectors into a LSTM Network. Can switch this to a more general w2v model 
    of google news or wikipedia
    
    Dataset: IMDB dataset of train and text files (can be converted to csvs of each eventually)

    
    System Setup:
    
    Ram of 8 and more 
    GPU (should be setup with cuda instalation and theano access to use)
    OS independent
    lib requried - pandas,sklearn,numpy,gensim,keras(theano backended)
	
    Training:
	Epoch 1/2: loss: 0.4437 - acc: 0.7917 - val_loss: 0.3403 - val_acc: 0.8510
	Epoch 2/2: loss: 0.2803 - acc: 0.8849 - val_loss: 0.3220 - val_acc: 0.8609
     
     Evaluate:
    
	Evaluate...
	25000/25000 [==============================] - 320s     
	('Test score:', 0.32195980163097382)
	('Test accuracy:', 0.86087999999999998)
		
    Results after 2 epochs:
        Validation Accuracy: 0.8485
                       Loss: 0.3442
'''


import multiprocessing
import numpy as np
import os
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
import glob
import codecs
from pandas import DataFrame
import pickle

# For Reproducibility
np.random.seed(1337)

# set parameters:
vocab_dim = 300
maxlen = 100
n_iterations = 10  # ideally more, since this improves the quality of the word vecs
n_exposures = 30
window_size = 7
batch_size = 32
n_epoch = 2 # can be increased to improe accuracy, not too much can lead to overfitting
input_length = 100
cpu_count = multiprocessing.cpu_count()

# label the movie review data source as positive(1) or negative(0)
SOURCES = [
    ('..\MoviePosNeg\\neg\*.txt',  0),
    ('..\MoviePosNeg\\pos\*.txt',  1)
]

TESTSOURCE = [
    ('..\MoviePosNeg\\test\\neg\*.txt',  0),
    ('..\MoviePosNeg\\test\\pos\*.txt',  1)
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
########### build data frame 
def build_data_frame(path, classification):
    rows = []
    index = []
    #print path
    for file_name, text in read_files(path):        
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame
       

############################################
########### Simple Parser converting each document to lower-case
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace 
        Extra cleaning can be done here Words become integers
            
    '''
    text = [document.lower().replace('\n', '').split() for document in text]
    return text


def create_dictionaries(train = None,
                        test = None,
                        model = None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (train is not None) and (model is not None) and (test is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(data):
            #import pdb;pdb.set_trace()            
            txt = data.lower().replace('\n', '').split()
            new_txt = []
            for word in txt:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            return new_txt
            #return data
            
        # read the movie review data
        testData['text'] =  test['text'].apply(parse_dataset) 
        trainData['text'] =  train['text'].apply(parse_dataset) 

        return w2indx, w2vec, trainData, testData
    else:
        print('No data provided...')

def train():

    print('Loading Data...') 
    testData = DataFrame({'text': [], 'class': []})
    for path, classification in TESTSOURCE:
        testData = testData.append(build_data_frame(path, classification))
    trainData = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        trainData = trainData.append(build_data_frame(path, classification))
    #data = data.reindex(numpy.random.permutation(data.index))
    print "Data Read"
    #train, test = import_tag(datasets = data_locations)
    combined = list(testData['text'].values) + list(trainData['text'].values)
    
    
    print('Tokenising...')
    combined = tokenizer(combined)
    
    
    #training word2vec on given dataset, possibly can use a pre built model of google news corpus
    print('Training a Word2vec model...')
    
    model = Word2Vec(size = vocab_dim,
                     min_count = n_exposures,
                     window = window_size,
                     workers = cpu_count,
                     iter = n_iterations)
    model.build_vocab(combined)
    model.train(combined)
    
    
    
    print('Transform the Data...')
    index_dict, word_vectors, train, test = create_dictionaries(train = trainData,
                                                                test = testData,
                                                                model = model)
    
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
    
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    
    print('Creating Datesets...')
    X_train = train['text']
    y_train = train['class']
    X_test = test['text']
    y_test = test['class']
    
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen = maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen = maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print('Convert labels to Numpy Sets...')
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print('Defining a Simple Sequential Keras Model...')
    model = Sequential()
    model.add(Embedding(output_dim = vocab_dim,
                        input_dim = n_symbols,
                        mask_zero = True,
                        weights = [embedding_weights],
                        input_length = input_length))
    
    model.add(LSTM(vocab_dim))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    
    print('Compiling the Model...')
    model.compile(optimizer = 'rmsprop',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    print("Train...")
    model.fit(X_train, y_train,
              batch_size = batch_size,
              nb_epoch = n_epoch,
              validation_data = (X_test, y_test),
              shuffle = True)
    
    print("Evaluate...")
    score = model.evaluate(X_test, y_test,
                           batch_size = batch_size)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print ('Saving parameters')
    json_string = model.to_json()
    model.save_weights('../ModelParams/LSTMparams.h5',overwrite=True)
    with open('../ModelParams/LSTMArch.p', 'wb') as f:
        pickle.dump(json_string, f)
    with open('../ModelParams/W2VModel.p', 'wb') as f:
        pickle.dump((index_dict, word_vectors), f)
    

def predict(sentences):
    from keras.models import model_from_json
    json_string = pickle.load(open('../ModelParams/LSTMArch.p','rb'))
    model = model_from_json(json_string)
    #model.compile
    model.load_weights('../ModelParams/LSTMparams.h5')
    w2indx, w2vec =  pickle.load(open('../ModelParams/W2VModel.p','rb'))
    def parse_dataset(data):
        #import pdb;pdb.set_trace()            
        txt = data.lower().replace('\n', '').split()
        new_txt = []
        for word in txt:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        return new_txt
    testData = [parse_dataset(x) for x in sentences] 
    model.compile
    #import pdb;pdb.set_trace() 
    X_train = sequence.pad_sequences(testData, maxlen = maxlen)
    p = model.predict(X_train)
    return p
    #import pdb;pdb.set_trace() 
    
if __name__ == '__main__':
    
    #call this function to train the function, comment it out if you just want predict    
    train()
    
    #pass a list of sentences which we need to classify
    listOfSentences = ['This is a horrible day','This is a beautiful day']    
    predictArray = predict(listOfSentences)

