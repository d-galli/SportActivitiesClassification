import pandas
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


DIGITS = 5
ACTIVITIES = ["sitting", "standing", "lying on back", "laying on right side","ascending stairs", "descending stairs", "standing in an elevator still",
              "moving around in an elevator", "walking in a parking lot", "walking on a treadmill (4 km/h, flat)",
              "walking on a treadmill (4 km/h, inclined)", "running on a treadmill (8 km/h)",
              "exercising on a stepper", "exercising on a cross trainer", "cycling on an exercise bike (horizontal)",
              "cycling on an exercise bike (vertical)", "rowing", "jumping", "playing basketball"]

print("Importing data ...")

train_data = pandas.read_csv("../sportsDataset/TrainingDataset.csv")
test_data = pandas.read_csv("../sportsDataset/TestDataset.csv")

train_data_input = train_data.loc[:, train_data.columns != 'activity_index']
train_data_output = train_data.loc[:, train_data.columns == 'activity_index'].values.ravel()
test_data_input = test_data.loc[:, test_data.columns != 'activity_index']
test_data_output = test_data.loc[:, test_data.columns == 'activity_index'].values.ravel()

scaler = StandardScaler()
scaler.fit(train_data_input)

train_data_input = scaler.transform(train_data_input)
test_data_input = scaler.transform(test_data_input)

train_data_output_dummy = pandas.get_dummies(train_data_output).rename(columns=lambda x: 'Category_' + str(x))
test_data_output_dummy = pandas.get_dummies(test_data_output).rename(columns=lambda x: 'Category_' + str(x))

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
model.fit(train_data_input, train_data_output_dummy, batch_size = 64, epochs = 200)

train_time = time.time()
print("Model trained:", round(train_time - start_time, DIGITS), "seconds")

print("Testing...")
accuracy = model.evaluate(test_data_input, test_data_output_dummy)[1]
predictions = model.predict(test_data_input, verbose=1).argmax(axis=-1) + 1
test_time = time.time()
print("Model tested:", round(test_time - train_time, DIGITS), "seconds")

print("Test is done. \nThe accuracy for this model is: ", round(accuracy, DIGITS))
print(f"The accuracy for this model is {round(accuracy, DIGITS)}.")
print("The classification report is:\n")
print(classification_report(test_data_output, predictions))
print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=ACTIVITIES)
fig, ax = plt.subplots(figsize=(15,15))
cmd.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
plt.tight_layout()
plt.savefig("MLP_confusion_matrix.png", pad_inches=100)
print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")
