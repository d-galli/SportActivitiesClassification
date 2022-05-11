import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


DIGITS = 5
CV_FOLDS = 7

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

# Scale the data so that each one can be uniformly evalauted
#Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated. Wikipedia explains the reasoning pretty well:
#Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
#The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.
scaler = StandardScaler()
scaler.fit(train_data_input)

train_data_input = scaler.transform(train_data_input)
test_data_input = scaler.transform(test_data_input)

print("Data imported")
#Train the model
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

print("Estimating accuracy ...")
correct = 0
total = 0
for i in range(len(predictions)):
    if predictions[i] == test_data_output[i]:
        correct += 1
    total += 1
accuracy = correct / total

print("Test is done. \nThe accuracy for this model is: ", round(accuracy, DIGITS))
print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")
print("The classification report is:\n")
print(classification_report(test_data_output, predictions))
print("Confusion matrix:")
print(confusion_matrix(test_data_output, predictions))

# Cross Validation

print("Performing Cross Validation ...")
knn_cv = neighbors.KNeighborsClassifier(n_neighbors = 19, metric = 'minkowski', p = 2)
cv_scores = cross_val_score(knn_cv, train_data_input, train_data_output, cv = 7)
print("Considering ", CV_FOLDS, " randomly created  groups and performing the cross validation, the accuracy values obtained are:\n", cv_scores)
print('which lead to a mean value of: {}'.format(round(np.mean(cv_scores)), DIGITS))

# Hypertuning of model parameters
print("Optimising model parameters ...")
knn_opt = neighbors.KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 36)}
print("Parameters grid created")
knn_gscv = GridSearchCV(knn_opt, param_grid, cv = CV_FOLDS)

print("Training the optimal model ...")
knn_gscv.fit(train_data_input, train_data_output)

print("Cross Validating the optimal model ...")
print("The optimal number of groups is: ",knn_gscv.best_params_["n_neighbors"])
print("The accuracy obtained with the best performing parameters is: ", round(knn_gscv.best_score_, DIGITS))



error = []

# Calculating error for K values between 1 and 40
for i in range(1, 20):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_data_input, train_data_output)
    pred_i = knn.predict(test_data_input)
    error.append(np.mean(pred_i != test_data_output))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()