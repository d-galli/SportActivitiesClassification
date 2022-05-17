import pandas as pd
from sklearn import decomposition
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

DIGITS = 5
ACTIVITIES = ["sitting", "standing", "lying on back", "laying on right side",
              "ascending stairs", "descending stairs", "standing in an elevator still",
              "moving around in an elevator", "walking in a parking lot",
              "walking on a treadmill (4 km/h, flat)",
              "walking on a treadmill (4 km/h, inclined)", "running on a treadmill (8 km/h)",
              "exercising on a stepper", "exercising on a cross trainer",
              "cycling on an exercise bike (horizontal)",
              "cycling on an exercise bike (vertical)", "rowing", "jumping",
              "playing basketball"]


def get_splitted_dataset(path):
    dataset = pd.read_csv(path)
    data_input = dataset.loc[:, dataset.columns != 'activity_index']
    data_output = dataset.loc[:, dataset.columns == 'activity_index'].values.ravel()
    return data_input, data_output


def get_principal_components(data_input, n_components=2):
    pca = decomposition.PCA(n_components=n_components)
    return pca.fit_transform(data_input)


def create_predictions_scatterplot(path, x, y, predictions):
    predictions_text = [ACTIVITIES[i - 1] for i in predictions]

    plt.figure(figsize=(15, 10))

    sns.set(style="darkgrid")
    sns.scatterplot(x=x, y=y, hue=predictions_text, palette="deep")

    plt.title('Predicted Classes', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.savefig(path)


def create_prediction_hits_scatterplot(path, x, y, data_output, predictions):
    prediction_correct = ['Correct' if predictions[i] == data_output[i]
                          else 'Wrong' for i in range(len(predictions))]

    plt.figure(figsize=(15, 10))

    sns.set(style="darkgrid")
    sns.scatterplot(x=x, y=y, hue=prediction_correct,
                    style=prediction_correct, palette="deep")

    plt.title('Classification Success', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.savefig(path)


def create_confusion_matrix_plot(path, confusion_matrix):
    cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=ACTIVITIES)
    fig, ax = plt.subplots(figsize=(15, 15))
    cmd.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.tight_layout()
    plt.savefig(path, pad_inches=100)