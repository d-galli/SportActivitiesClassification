import pandas
import time
from sklearn import neighbors

DIGITS = 5

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60

    time_elapsed = "{0}:{1}:{2}".format(int(hours),int(mins),round(sec, 3))
    return time_elapsed

print("Importing data ...")

train_data = pandas.read_csv("../sportsDataset/TrainingDataset.csv")
test_data = pandas.read_csv("../sportsDataset/TestDataset.csv")

train_data_input = train_data.loc[:, train_data.columns != 'activity_index']
train_data_output = train_data.loc[:, train_data.columns == 'activity_index'].values.ravel()
test_data_input = test_data.loc[:, test_data.columns != 'activity_index']
test_data_output = test_data.loc[:, test_data.columns == 'activity_index'].values.ravel()

print("Data imported")
print("Training ...")
start_time = time.time()
model = neighbors.KNeighborsClassifier(n_neighbors = 19, metric = 'minkowski', p = 2)
model.fit(train_data_input, train_data_output)
train_time = time.time()
print("Model trained:", round(train_time - start_time, DIGITS), "seconds")

print("Testing ...")
predictions = model.predict(test_data_input)
test_time = time.time()
print("Model tested:", round(test_time - train_time, DIGITS), "seconds")
print("Validating ...")
correct = 0
total = 0
for i in range(len(predictions)):
    if predictions[i] == test_data_output[i]:
        correct += 1
    total += 1
accuracy = correct / total

print("Test is done. \nThe accuracy for this model is: ", round(accuracy, DIGITS))
print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")