import pandas
import numpy as np
import time
import os.path
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from scikeras.wrappers import KerasClassifier

from sklearn.metrics import classification_report,\
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
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

train_data_output_dummy = \
    pandas.get_dummies(train_data_output).rename(columns=lambda x: 'Category_' + str(x))
test_data_output_dummy = \
    pandas.get_dummies(test_data_output).rename(columns=lambda x: 'Category_' + str(x))

print("Data imported")
print("Training ...")
start_time = time.time()

mlp = Sequential()
mlp.add(Dense(units = 40,
                kernel_initializer = 'uniform',
                activation = 'relu',
                input_dim = 90))
mlp.add(Dropout(rate = 0.2))
mlp.add(Dense(units = 40,
                kernel_initializer = 'uniform',
                activation = 'relu'))
mlp.add(Dropout(rate = 0.2))
mlp.add(Dense(units = 19,
                kernel_initializer = 'uniform',
                activation = 'softmax'))
mlp.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model = KerasClassifier(build_fn=mlp, epochs=100, batch_size=128, verbose=1)

model.fit(train_data_input, train_data_output_dummy)

train_time = time.time()
print(f"Model trained: {round(train_time - start_time, utils.DIGITS)} seconds")

print("Testing...")
predictions = model.predict(test_data_input, verbose=1).argmax(axis=-1) + 1
test_time = time.time()
print(f"Model tested: {round(test_time - train_time, utils.DIGITS)} seconds")

print(f"The overall time required by this model is: "
      f"{round(test_time - start_time, utils.DIGITS)} seconds")

print("The classification report:")
print(classification_report(test_data_output, predictions, zero_division=0))

print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
utils.create_confusion_matrix_plot("MLP_confusion_matrix.png", confusion_matrix)

print("Visualise decision boundaries")
print("Performing PCA ...")
principal_components = utils.get_principal_components(test_data_input)
print("Done.")

print("Plotting predictions ...")
utils.create_predictions_scatterplot("MLP_predictions_scatterplot.png",
                                     principal_components[:, 0],
                                     principal_components[:, 1],
                                     predictions)

utils.create_prediction_hits_scatterplot("MLP_prediction_hits_scatterplot.png",
                                         principal_components[:, 0],
                                         principal_components[:, 1],
                                         test_data_output,
                                         predictions)
print("Done")

print("Cross Validation")
model = Sequential()
model.add(Dense(units = 40,
                kernel_initializer = 'uniform',
                activation = 'relu',
                input_dim = 90))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 40,
                kernel_initializer = 'uniform',
                activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 19,
                kernel_initializer = 'uniform',
                activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=128, verbose=0)
data_input = np.concatenate([train_data_input, test_data_input])
data_output = np.concatenate([train_data_output_dummy, test_data_output_dummy])
cv_scores = utils.get_cross_validation_score(estimator, data_input, data_output)
print("Done")
