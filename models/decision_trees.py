import pandas
import time
import pydotplus 
from sklearn import tree
from matplotlib import pyplot as plt

DIGITS = 5
ACTIVITIES = ["sitting", "standing", "lying on back", "laying on right side","ascending stairs", "descending stairs", "standing in an elevator still",
              "moving around in an elevator", "walking in a parking lot", "walking on a treadmill with a speed of 4 km/h on flat",
              "walking on a treadmill with a speed of 4 km/h on a 15 deg inclined ositions", "running on a treadmill with a speed of 8 km/h",
              "exercising on a stepper", "exercising on a cross trainer", "cycling on an exercise bike in horizontal position",
              "cycling on an exercise bike in vertical positions", "rowing", "jumping", "playing basketball"]

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
model = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 19)
model.fit(train_data_input, train_data_output)
train_time = time.time()
print("Model trained:", round(train_time - start_time, DIGITS), "seconds")
#print(time_convert(train_time - start_time))

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


dot_data = tree.export_graphviz(model, feature_names = train_data_input.columns, class_names = ACTIVITIES, filled = True, out_file = None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("decision_tree.pdf")