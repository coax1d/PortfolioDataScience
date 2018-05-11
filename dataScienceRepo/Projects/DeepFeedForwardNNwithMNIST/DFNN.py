
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import sgd
from keras.utils import np_utils
from keras import backend as K
import numpy as np

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

path_to_dataset = "./mnist.pkl.gz"

(X_train, y_train), (X_test, y_test) = mnist.load_data(path_to_dataset)

batch_size = 60
epochs = 10
classes = 10


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.np_utils.to_categorical(y_train, classes)
y_test = keras.utils.np_utils.to_categorical(y_test, classes)



model = Sequential()
model.add(Dense(50, activation='relu', input_dim=784))
model.add(Dense(275, activation= 'relu', input_dim = 50))
model.add(Dense(100, activation ='relu', input_dim = 275))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,verbose = 0)
train_score = model.evaluate(X_train,y_train,verbose=0)
train1 = train_score[1]
score = model.evaluate(X_test, y_test,verbose=0)
test1 = score[1]

model2 = Sequential()
model2.add(Dense(250, activation='relu', input_dim=784))
model2.add(Dense(275, activation= 'relu', input_dim = 250))
model2.add(Dense(100, activation ='relu', input_dim = 275))
model2.add(Dense(10, activation='softmax'))
model2.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model2.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,verbose = 0)
train_score = model2.evaluate(X_train,y_train,verbose=0)
train2 = train_score[1]
score = model2.evaluate(X_test, y_test,verbose=0)
test2 = score[1]




model3 = Sequential()
model3.add(Dense(50, activation='relu', input_dim=784,
                 bias_regularizer=keras.regularizers.l2(0.03)))
model3.add(Dense(275, activation= 'relu', input_dim = 50))
model3.add(Dense(100, activation ='relu', input_dim = 275))
model3.add(Dense(300, activation = 'relu', input_dim = 100))
model3.add(Dense(275, activation = 'relu', input_dim = 300))
model3.add(Dense(10, activation='softmax'))
model3.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model3.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,verbose=0)
train_score = model3.evaluate(X_train,y_train,verbose=0)
train3 = train_score[1]
score = model3.evaluate(X_test, y_test,verbose=0)
test3 = score[1]

model4 = Sequential()
model4.add(Dense(250, activation='relu', input_dim=784,
                 bias_regularizer=keras.regularizers.l2(0.03)))
model4.add(Dense(275, activation= 'relu', input_dim = 250))
model4.add(Dense(100, activation ='relu', input_dim = 275))
model4.add(Dense(300, activation = 'relu', input_dim = 100))
model4.add(Dense(275, activation = 'relu', input_dim = 300))
model4.add(Dense(10, activation='softmax'))
model4.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model4.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,verbose=0)
train_score = model4.evaluate(X_train,y_train,verbose = 0)
train4 = train_score[1]
score = model4.evaluate(X_test, y_test,verbose =0)
test4 = score[1]


print("Accuracies")
print("3 Hidden layers, first HLN50+no regularization Train/Test: ", train1, test1)
print("3 Hidden layers, first HLN250+ regularization Train/Test:  ", train3, test3)
print("5 Hidden layers, first HLN50+no regularization Train/Test: ", train2, test2)
print("5 Hidden layers, first HLN250+ regularization Train/Test: ", train4, test4)
print("Used 10 epochs, l2 regularization with .03 as the learning rate, used Schotastic Gradient descent," +
     " by using deep learning got an increase of up to 6 percent to accuracy")











