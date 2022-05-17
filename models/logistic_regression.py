import time

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import utils

print("Importing data ...")

train_data_input, train_data_output = utils.get_splitted_dataset("../sportsDataset/TrainingDataset.csv")
test_data_input, test_data_output = utils.get_splitted_dataset("../sportsDataset/TestDataset.csv")

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
print(f"Model trained: {round(train_time - start_time, utils.DIGITS)} seconds")

print("Computing predictions ...")
predictions = [int(i) for i in model.predict(test_data_input)]
test_time = time.time()
print(f"Model tested: {round(test_time - train_time, utils.DIGITS)} seconds")

print("The classification report is:\n")
print(classification_report(test_data_output, predictions, zero_division=0))

print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
utils.create_confusion_matrix_plot("LR_confusion_matrix.png", confusion_matrix)

print("Visualise decision boundaries")
print("Performing PCA ...")
principal_components = utils.get_principal_components(test_data_input)
print("Done.")

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
print("Done.")

print(f"The overall time reqired by this model is: {round(test_time - start_time, utils.DIGITS)} seconds")
