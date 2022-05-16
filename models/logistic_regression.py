import pandas
import time

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

print("Data imported")
print("Training ...")
start_time = time.time()
model = linear_model.LogisticRegressionCV(max_iter=8000, cv=7)
model.fit(train_data_input, train_data_output)
train_time = time.time()
print("Model trained:", round(train_time - start_time, DIGITS), "seconds")

print("Computing predictions ...")
predictions = model.predict(test_data_input)
test_time = time.time()
print("Model tested:", round(test_time - train_time, DIGITS), "seconds")

print("The classification report is:\n")
print(classification_report(test_data_output, predictions, zero_division=0))
print("")
print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=ACTIVITIES)
fig, ax = plt.subplots(figsize=(15,15))
cmd.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
plt.tight_layout()
plt.savefig("LR_confusion_matrix.png", pad_inches=100)
print("")
print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")
