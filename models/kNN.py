import time
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from utils import utils

print("Importing data ...")

train_data_input, train_data_output = \
    utils.get_splitted_dataset(
        os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),
                     "../sportsDataset/TrainingDataset.csv")
    )
test_data_input, test_data_output = \
    utils.get_splitted_dataset(
        os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),
                     "../sportsDataset/TestDataset.csv")
    )

# Scale the data so that each one can be uniformly evalauted
# Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated. Wikipedia explains the reasoning pretty well:
# Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
# The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.

print("Scaling data ...")
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
print(f"Model trained: {round(train_time - start_time, utils.DIGITS)} seconds")

print("Testing ...")
predictions = [int(i) for i in model.predict(test_data_input)]
test_time = time.time()
print(f"Model tested: {round(test_time - train_time, utils.DIGITS)} seconds")

print(f"The overall time required by this model is: "
      f"{round(test_time - start_time, utils.DIGITS)} seconds")

print("The classification report is:")
print(classification_report(test_data_output, predictions))

print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
utils.create_confusion_matrix_plot("kNN_confusion_matrix.png", confusion_matrix)


print("Visualise decision boundaries")
print("Performing PCA ...")
principal_components = utils.get_principal_components(test_data_input)
print("Done.")

print("Plotting predictions ...")
utils.create_predictions_scatterplot("kNN_predictions_scatterplot.png",
                                     principal_components[:, 0],
                                     principal_components[:, 1],
                                     predictions)

utils.create_prediction_hits_scatterplot("kNN_prediction_hits_scatterplot.png",
                                         principal_components[:, 0],
                                         principal_components[:, 1],
                                         test_data_output,
                                         predictions)

# Cross Validation

print("Cross Validation")
knn_cv = neighbors.KNeighborsClassifier(n_neighbors = 19,
                                        metric = 'minkowski',
                                        p = 2)
cv_scores = cross_val_score(knn_cv,
                            train_data_input,
                            train_data_output,
                            cv = utils.CV_FOLDS)
print(f"Considering {utils.CV_FOLDS} randomly created  groups "
      f"and performing the cross validation, the accuracy values "
      f"obtained are: \n {cv_scores}")
print(f"which lead to a mean value of: {round(np.mean(cv_scores), utils.DIGITS)}")

# Hypertuning of model parameters
print("Optimising model parameters ...")
knn_opt = neighbors.KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 20)}
print("Parameters grid created")
knn_gscv = GridSearchCV(knn_opt, param_grid, cv = utils.CV_FOLDS)

print("Training the optimal model ...")
knn_gscv.fit(train_data_input, train_data_output)

print("Cross Validating the optimal model ...")
print(f"The optimal number of groups is: {knn_gscv.best_params_['n_neighbors']}")
print(f"The accuracy obtained with the best performing "
      f"parameters is: {round(knn_gscv.best_score_, utils.DIGITS)}")

error = []

for i in range(1, 20):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_data_input, train_data_output)
    pred_i = knn.predict(test_data_input)
    error.append(np.mean(pred_i != test_data_output))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value', fontsize = 20)
plt.xlabel('K Value' , fontsize = 15)
plt.ylabel('Mean Error', fontsize = 15)
plt.savefig("kNN_error_rate_k_value_plot.png")

print("Done")