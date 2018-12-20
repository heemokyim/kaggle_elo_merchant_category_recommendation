# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:48:10 2018

@author: bcheung
"""
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

class denoise_autoencoder():
    
    def __init__(self,encoding_dim,encoding_layers,epochs,
                 batch_size,shuffle,optimizer='adam',loss='mse',
                 input_swap_noise=0.15=,test_size=0,scale=False):
        self.encoding_dim = encoding_dim
        self.encoding_layers = encoding_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.loss = loss
        self.input_swap_noise = input_swap_noise
        self.test_size = test_size
        self.scale = scale
        
    def autoencoder_setup(self,ncol):
        
        input_dim = Input(shape=(ncol,))
        encoded1 = Dense(1028, activation = 'relu')(input_dim)
        encoded2 = Dense(512, activation = 'relu')(encoded1)
        encoded3 = Dense(256, activation = 'relu')(encoded2)
        self.encoder = Dense(self.encoding_dim, activation = 'relu')(encoded3)
        
        # Decoder Layers
        decoded1 = Dense(256, activation = 'relu')(self.encoder)
        decoded2 = Dense(512, activation = 'relu')(decoded1)
        decoded3 = Dense(1028, activation = 'relu')(decoded2)
        decoded = Dense(ncol, activation = 'sigmoid')(decoded3)
        
        self.autoencoder = Model(inputs = input_dim, outputs = decoded)
        self.autoencoder.compile(optimizer = self.optimizer, loss = self.loss)
        
        self.encoder = Model(inputs = input_dim, outputs = self.encoder)
    
    def fit(self,X):
        
        self.autoencoder_setup(X.shape[1])
        
        if self.scale:
            X = minmax_scale(X,axis=0)
        
        if self.test_size>0:
            X_train, X_test = train_test_split(X,test_size=self.test_size,random_state=42)
            self.autoencoder.fit(X_train, X_train, 
                                epochs = self.epochs, 
                                batch_size = self.batch_size, 
                                shuffle = self.shuffle, 
                                validation_data = [X_test, X_test])
        else:
            self.autoencoder.fit(X, X, 
                                epochs = self.epochs, 
                                batch_size = self.batch_size, 
                                shuffle = self.shuffle)
    
    
    def predict(self,X):
        
        return(pd.DataFrame(self.encoder.predict(X)))

if __name__ == '__main__':
    dae = denoise_autoencoder(encoding_dim=200,
                          encoding_layers=3,
                          epochs=15,
                          batch_size=80,
                          shuffle=True,
                          optimizer='adam',
                          loss='mse',
                          test_size=0.2,
                          scale=True)
    dae.fit(X)
    encoded_X = dae.predict(X)
