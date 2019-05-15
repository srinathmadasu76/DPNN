# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:35:23 2018

@author: hb65402
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:56:48 2018

@author: Srinath
"""
import os
from keras import backend as K
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,UpSampling2D, Conv1D, MaxPooling1D,UpSampling1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12345)

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
np.random.seed(0)
tf.set_random_seed(0)

#matplotlib inline
def Autoencoder():
    global np
    tf.reset_default_graph()
    init = tf.initialize_all_variables()
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)
    sess.run(init)
    K.set_session(sess)

    def makeXy(ts, nb_timesteps):
        """
        Input: 
               ts: original time series
               nb_timesteps: number of time steps in the regressors
        Output: 
               X: 2-D array of regressors
               y: 1-D array of target 
        """
        X = []
        y = []
        for i in range(nb_timesteps, ts.shape[0]):
            X.append(list(ts.loc[i-nb_timesteps:i-1]))
            y.append(ts.loc[i])
        X, y = np.array(X), np.array(y)
        return X, y
    res = pd.read_csv('result.csv',header=0)
    #res=res[['Predictions']]
    res=res[['Actual']]
    df = pd.read_csv('result.csv',header=0)
    #df[['v']] =  res
    df=df[['Actual']]
    datasetinput = df.values
    
    X_train, X_test = train_test_split(datasetinput, test_size=0.2,shuffle=False)
    
    X_train = X_train.reshape(X_train.shape[0],1)
    
    X_test = X_test.reshape(X_test.shape[0],1)
    print(np.shape(X_train))
    df1 = pd.read_csv('AE.csv',header=0)
    df1 = df1[['v']]
    datasetoutput = res.values
    
    Y_train, Y_test = train_test_split(datasetoutput, test_size=0.2,shuffle=False)
    
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    
    #input_layer = Input(shape=(80,2))
    #encoder
    
    #model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(Dense(100, activation='relu'))
    
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    #conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    
        #decoder
    #conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    #up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    #conv5 = Conv1D(32, (3), activation='relu', padding='same')(conv2) # 14 x 14 x 64
    #up2 = UpSampling1D((2))(conv5) # 28 x 28 x 64
    #decoder = Conv1D(1, (3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    
    #encoder = Dense(encoding_dim, activation="tanh", 
    #                activity_regularizer=regularizers.l1(10e-5))(input_layer)
    #encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    #
    #decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    #decoder = Dense(1, activation='relu')(decoder)
    
    #autoencoder = Model(inputs=input_layer, outputs=decoder)
    from keras.models import Model
    from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
    import numpy as np
    
    inp =  Input(shape=(1,1))
    
    encoded = Dense(1, activation='linear')(inp)
    encoded = Dense(6, activation='sigmoid')(encoded)
    #flat = Flatten()(encoded)
    encoded = Dense(6, activation='relu')(encoded)
    #encoded = Dense(1, activation='linear')(flat)
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))#8
    model.add(Dense(10, activation='relu'))#10
    model.add(Dense(1, activation='linear'))
    model.add(Dense(10, activation='relu'))#10
    model.add(Dense(8, activation='sigmoid'))#8
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    
    decoded = Dense(1, activation='relu')(encoded)
    decoded = Dense(6, activation='relu')(decoded)
    decoded = Dense(12, activation='relu')(encoded)
    flat = Flatten()(encoded)
    decoded = Dense(1, activation='linear')(flat)
    
    ##conv = Conv1D(filters=64, kernel_size=1)(inp)
    ##pool = MaxPool1D(pool_size=1)(conv)
    #conv1 = Conv1D(filters=64, kernel_size=1)(pool)
    #flat = Flatten()(pool)
    #dense = Dense(1)(flat)
    
    ##x2_ = Conv1D(32, 1, activation='relu', padding='valid')(pool)
    ##x1_ = UpSampling1D(1)(x2_)
    #x_ = Conv1D(64, 3, activation='relu', padding='valid')(x1_)
    #upsamp = UpSampling1D(2)(x_)
    ##flat = Flatten()(x1_)
    #decoded = Dense(1,activation = 'relu')(flat)
    ##decoded = Conv1D(1, 1, activation='relu', padding='same')(flat)
    #decoded = Dense(1,activation = 'relu')(flat)
    #model = Model(inp, decoded)
    model.compile(loss='mse', optimizer='adam')
    
    print(model.summary())
    
    # fit model
    model.fit(X_train, Y_train, epochs=500)
    
    
    predictions = model.predict(X_test)
    print(predictions)
    mse = np.mean(np.power(Y_test - predictions, 2), axis=1)
    print("MSE=",mse)       
    np.savetxt("result2.csv",np.c_[Y_test,predictions],fmt=['%0.5f','%0.5f'],delimiter=',',header="Actual,Predictions",comments='')
#CLOSE TF SESSION
    K.clear_session()
    sess.close()