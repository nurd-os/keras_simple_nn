import keras
from keras import Sequential
from keras.layers.core import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(0, 100)
y = np.array([0] * 20 + [1] * 60 + [0] * 20)

sigmoid = lambda x: 1/(1+np.exp(-x))
relu = lambda x: x*(x>=0)

def exe():

    model = Sequential()

    model.add(Dense(2, activation='relu', input_dim=1))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x.reshape(-1,1), y, epochs=100)


    w1 = model.get_weights()[0]
    b1 = model.get_weights()[1]

    w2 = model.get_weights()[2]
    b2 = model.get_weights()[3]

    layer_1 = np.dot(x.reshape(-1,1), w1) + b1
    layer_2 = np.dot(relu(layer_1), w2) + b2

    plt.scatter(x, y)
    plt.plot(x, layer_2)
    plt.show(block=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    exe()

