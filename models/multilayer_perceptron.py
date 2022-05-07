import pandas
import time
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

DIGITS = 5

print("Importing data ...")

train_data = pandas.read_csv("../sportsDataset/TrainingDataset.csv")
test_data = pandas.read_csv("../sportsDataset/TestDataset.csv")

train_data_input = train_data.loc[:, train_data.columns != 'activity_index']
train_data_output = train_data.loc[:, train_data.columns == 'activity_index'].values.ravel()
test_data_input = test_data.loc[:, test_data.columns != 'activity_index']
test_data_output = test_data.loc[:, test_data.columns == 'activity_index'].values.ravel()

train_data_output = pandas.get_dummies(train_data_output).rename(columns=lambda x: 'Category_' + str(x))
test_data_output = pandas.get_dummies(test_data_output).rename(columns=lambda x: 'Category_' + str(x))

print("Data imported")
print("Training ...")
start_time = time.time()

model = Sequential()
model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 90))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(train_data_input, train_data_output, batch_size = 10, epochs = 100)

train_time = time.time()
print("Model trained:", round(train_time - start_time, DIGITS), "seconds")

print("Testing...")
accuracy = model.evaluate(test_data_input, test_data_output)[1]
test_time = time.time()
print("Model tested:", round(test_time - train_time, DIGITS), "seconds")

print("Test is done. \nThe accuracy for this model is: ", round(accuracy, DIGITS))
print(f"The accuracy for this model is {round(accuracy, DIGITS)}.")
print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")
