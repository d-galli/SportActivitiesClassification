import numpy as np
import time
import os.path
import sys

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import utils

print("Importing data ...")

parent_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
train_data_input, train_data_output = \
    utils.get_splitted_dataset(
        os.path.join(parent_dir, "../sports_dataset/training_dataset.csv")
    )
test_data_input, test_data_output = \
    utils.get_splitted_dataset(
        os.path.join(parent_dir, "../sports_dataset/test_dataset.csv")
    )

scaler = StandardScaler()
scaler.fit(train_data_input)

train_data_input = scaler.transform(train_data_input)
test_data_input = scaler.transform(test_data_input)

print("Data imported")
print("Training ...")
start_time = time.time()
model = linear_model.LogisticRegression(max_iter=8000)
model.fit(train_data_input, train_data_output)
train_time = time.time()
print(f"Model trained: {round(train_time - start_time, utils.DIGITS)} seconds")

print("Computing predictions ...")
predictions = [int(i) for i in model.predict(test_data_input)]
test_time = time.time()
print(f"Model tested: {round(test_time - train_time, utils.DIGITS)} seconds")

print(f"The overall time reqired by this model is: "
      f"{round(test_time - start_time, utils.DIGITS)} seconds")

print("The classification report is:\n")
print(classification_report(test_data_output, predictions, zero_division=0))

print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
utils.create_confusion_matrix_plot("LR_confusion_matrix.png", confusion_matrix)

print("Visualise decision boundaries")
print("Performing PCA ...")
principal_components = utils.get_principal_components(test_data_input)
print("Done")

print("Plotting predictions ...")
utils.create_predictions_scatterplot("LR_predictions_scatterplot.png",
                                     principal_components[:, 0],
                                     principal_components[:, 1],
                                     predictions)

utils.create_prediction_hits_scatterplot("LR_prediction_hits_scatterplot.png",
                                         principal_components[:, 0],
                                         principal_components[:, 1],
                                         test_data_output,
                                         predictions)
print("Done")

print("Cross Validation")
classifier_cv = linear_model.LogisticRegression(max_iter=8000)
data_input = np.concatenate([train_data_input, test_data_input])
data_output = np.concatenate([train_data_output, test_data_output])
cv_scores, mean_cv_scores = utils.get_cross_validation_score(classifier_cv, data_input, data_output)
print("Done")
