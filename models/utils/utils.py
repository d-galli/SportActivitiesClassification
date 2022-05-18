import os.path
import sys
import pandas as pd
from sklearn import decomposition, tree
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

PLOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "plots")
CV_FOLDS = 7
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


def create_predictions_scatterplot(filename, x, y, predictions):
    predictions_text = [ACTIVITIES[i - 1] for i in predictions]

    plt.figure(figsize=(15, 10))

    sns.set(style="darkgrid")
    sns.scatterplot(x=x, y=y, hue=predictions_text, palette="deep")

    plt.title('Predicted Classes', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    if not os.path.isdir(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

    plt.savefig(os.path.join(PLOT_DIRECTORY, filename))


def create_prediction_hits_scatterplot(filename, x, y, data_output, predictions):
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

    if not os.path.isdir(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

    plt.savefig(os.path.join(PLOT_DIRECTORY, filename))


def create_confusion_matrix_plot(filename, confusion_matrix):
    cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=ACTIVITIES)
    fig, ax = plt.subplots(figsize=(15, 15))
    cmd.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.tight_layout()

    if not os.path.isdir(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

    plt.savefig(os.path.join(PLOT_DIRECTORY, filename), pad_inches=100)


def create_impurity_vs_alpha_plot(filename, impurities, ccp_alphas):
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='.', drawstyle="steps-post")
    ax.set_xlabel('Effective $\\alpha $', fontsize=15)
    ax.set_ylabel("Total impurity of leaves", fontsize=15)
    ax.set_title('Total impurity VS Effective $\\alpha $', fontsize=20)
    plt.savefig(os.path.join(PLOT_DIRECTORY, filename))


def create_depth_vs_alpha_plot(filename, tree_depths, ccp_alphas):
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas[:-1], tree_depths[:-1], marker='.')
    plt.xlabel("Effective $ \\alpha$", fontsize=15)
    plt.ylabel("Depth of the tree", fontsize=15)
    plt.xscale("log")
    plt.title('Total depth  VS Effective $\\alpha $', fontsize=20)
    plt.savefig(os.path.join(PLOT_DIRECTORY, filename))


def create_accuracy_vs_alpha_plot(filename, acc_scores, ccp_alphas):
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(ccp_alphas[:-1], acc_scores[:-1], marker='.')
    plt.xlabel("Effective $ \\alpha$", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.xscale("log")
    plt.title('Accuracy VS Effective $\\alpha $', fontsize=20)
    plt.savefig(os.path.join(PLOT_DIRECTORY, filename))


def get_parametrized_decision_trees(criterion, param, param_values):
    return [tree.DecisionTreeClassifier(criterion=criterion, **{param: i}) for i in param_values]


def get_accuracies_of_decision_trees(dtrees, test_data_input, test_data_output):
    accuracies = []
    for dtree in dtrees:
        pred = dtree.predict(test_data_input)
        accuracies.append(accuracy_score(test_data_output, pred))
    return accuracies


def create_param_accuracy_plot(filename, param, data):
    plt.figure(figsize=(10, 6))
    plt.plot(param, 'acc_gini', data=data, label='gini')
    plt.plot(param, 'acc_entropy', data=data, label='entropy')
    plt.xlabel(param)
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIRECTORY, filename))

def get_parametrized_decision_tree_accuracies(param,
                                              param_values,
                                              train_data_input,
                                              train_data_output,
                                              test_data_input,
                                              test_data_output):
    accuracies = {}
    for criterion in ["gini", "entropy"]:
        decision_trees = get_parametrized_decision_trees("gini", param, param_values)
        for dtree in decision_trees:
            dtree.fit(train_data_input, train_data_output)
        accuracies[f"acc_{criterion}"] = \
            get_accuracies_of_decision_trees(decision_trees,
                                             test_data_input,
                                             test_data_output)

    return accuracies