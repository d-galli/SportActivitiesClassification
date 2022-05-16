import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import decomposition


DIGITS = 6
CV_FOLDS = 7

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
print("Model trained:", round(train_time - start_time, DIGITS), "seconds")

print("Testing ...")
predictions = model.predict(test_data_input)
test_time = time.time()
print("Model tested:", round(test_time - train_time, DIGITS), "seconds")

print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")
print("The classification report is:\n")
print(classification_report(test_data_output, predictions))
print("Confusion matrix:")
print(confusion_matrix(test_data_output, predictions))

print("Visualise decision boundaries")
print("Performing PCA ...")
pca = decomposition.PCA(n_components = 2) # 2D representation requires 2 components
principalComponents = pca.fit_transform(test_data_input)

activitiesDictionary = {1:"sitting", 2:"standing", 3:"lying on back", 4:"laying on right side", 5:"ascending stairs", 6:"descending stairs", 7:"standing in an elevator still",
              8:"moving around in an elevator", 9:"walking in a parking lot", 10:"walking (flat treadmill)", 
              11:"walking (inclined treadmill)", 12:"running (flat treadmill)",
              13:"exercising on a stepper", 14:"cross training", 15:"cycling (horizontal position)",
              16:"cycling (vertical position)", 17:"rowing", 18:"jumping", 19:"playing basketball"}


predictionsText = []

for i in range(predictions.size):
    item = predictions[i]
    predictedClass = activitiesDictionary[item]
    predictionsText.append(predictedClass)

print("Plotting predictions ...")

plt.figure(figsize = (15,10))

sns.set(style="darkgrid")
sns.scatterplot(x = principalComponents[:, 0], y = principalComponents[:, 1], hue = predictionsText, palette = "deep")

plt.title('Predicted Classes', fontsize = 20)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.legend(bbox_to_anchor=(1.02, 1), loc = 2, borderaxespad = 0.)
plt.subplots_adjust(right = 0.75)
#plt.show()

predictionCorrect = []
for i in range(len(predictions)):
    if predictions[i] == test_data_output[i]:
        predictionCorrect.append('Correct')
    
    else:
        predictionCorrect.append('Wrong')

plt.figure(figsize = (15,10))

sns.set(style="darkgrid")
sns.scatterplot(x = principalComponents[:, 0], y = principalComponents[:, 1], hue = predictionCorrect, style = predictionCorrect, palette = "deep")

plt.title('Classification Success', fontsize = 20)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.legend(bbox_to_anchor=(1.02, 1), loc = 2, borderaxespad = 0.)
plt.subplots_adjust(right = 0.75)
#plt.show()

# Cross Validation

print("Cross Validation")
knn_cv = neighbors.KNeighborsClassifier(n_neighbors = 19, metric = 'minkowski', p = 2)
cv_scores = cross_val_score(knn_cv, train_data_input, train_data_output, cv = CV_FOLDS)
print("Considering ", CV_FOLDS, " randomly created  groups and performing the cross validation, the accuracy values obtained are:\n", cv_scores)
print('which lead to a mean value of: {}'.format(round(np.mean(cv_scores), DIGITS)))

# Hypertuning of model parameters
print("Optimising model parameters ...")
knn_opt = neighbors.KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 20)}
print("Parameters grid created")
knn_gscv = GridSearchCV(knn_opt, param_grid, cv = CV_FOLDS)

print("Training the optimal model ...")
knn_gscv.fit(train_data_input, train_data_output)

print("Cross Validating the optimal model ...")
print("The optimal number of groups is: ",knn_gscv.best_params_["n_neighbors"])
print("The accuracy obtained with the best performing parameters is: ", round(knn_gscv.best_score_, DIGITS))

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
plt.show()