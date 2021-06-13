import pandas as pd
from sklearn import preprocessing
import numpy as np

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import matplotlib.pyplot as plt
np.random.seed(4)
tf.random.set_seed(4)

history_points = 50

def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('Date', axis=1)
    data = data.drop(0, axis=0)

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

     # using the last {history_points} open high low close volume data points, predict the next open value
    ohlcv_histories_normalised = np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[:,0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data.iloc[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

def main():
    print("Hello World!")
    ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('SPY5YDaily.csv')

    test_split = 0.9 # the percent of data to be used for testing
    n = int(ohlcv_histories.shape[0] * test_split)

    # splitting the dataset up into train and test sets

    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)

    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='mse')

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')

    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(ohlcv_test, y_test)
    print(evaluation)

    y_test_predicted = model.predict(ohlcv_test)
    # model.predict returns normalised values
    # now we scale them back up using the y_scaler from before
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    # also getting predictions for the entire dataset, just to see how it performs
    y_predicted = model.predict(ohlcv_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()


if __name__ == "__main__":
    main()