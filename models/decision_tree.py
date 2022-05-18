import pandas as pd
import time
import pydotplus
import numpy as np
import os.path
import sys

from sklearn import tree, decomposition
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
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
print("Scaling data ...")

scaler = StandardScaler()
scaler.fit(train_data_input)
train_data_input = pd.DataFrame(scaler.transform(train_data_input),
                                index=train_data_input.index,
                                columns=train_data_input.columns)
test_data_input = pd.DataFrame(scaler.transform(test_data_input),
                                index=test_data_input.index,
                                columns=test_data_input.columns)
print("Data imported")

print("Training ...")
start_time = time.time()
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=19)
model.fit(train_data_input, train_data_output)
train_time = time.time()
print("Model trained:", round(train_time - start_time, utils.DIGITS), "seconds")

print("Testing ...")
predictions = [int(i) for i in model.predict(test_data_input)]
test_time = time.time()
print("Model tested:", round(test_time - train_time, utils.DIGITS), "seconds")

print("The overall time reqired by this model is: ", round(test_time - start_time, utils.DIGITS), "seconds")

print("The classification report is:\n")
print(classification_report(test_data_output, predictions, zero_division=0))

print("Confusion matrix:")
confusion_matrix = confusion_matrix(test_data_output, predictions)
print(confusion_matrix)
utils.create_confusion_matrix_plot("DT_confusion_matrix.png", confusion_matrix)

print("Exporting the tree ...")
dec_tree_data = tree.export_graphviz(model,
                                     feature_names=train_data_input.columns,
                                     class_names=utils.ACTIVITIES,
                                     filled=True,
                                     out_file=None)
graph = pydotplus.graph_from_dot_data(dec_tree_data)
graph.write_png(os.path.join(utils.PLOT_DIRECTORY, "DT_structure.png"))
print('Tree saved as PNG')

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
dt_cv = tree.DecisionTreeClassifier(criterion='entropy', max_depth=19)
cv_scores = cross_val_score(dt_cv, train_data_input, train_data_output, cv=utils.CV_FOLDS)
print(f"Considering {utils.CV_FOLDS} randomly created groups and performing "
      f"the cross validation, the accuracy values obtained are:\n {cv_scores}")
print(f"which lead to a mean value of: {round(np.mean(cv_scores), utils.DIGITS)}")

print("Optimising model parameters ...")

# optimizing using effective alpha
path = model.cost_complexity_pruning_path(train_data_input, train_data_output)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
for ccp_alpha in ccp_alphas:
    dec_tree_opt = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dec_tree_opt.fit(train_data_input, train_data_output)
    clfs.append(dec_tree_opt)

# plotting
utils.create_impurity_vs_alpha_plot("DT_impurity_vs_alpha.png", impurities, ccp_alphas)

tree_depths = [dec_tree_opt.tree_.max_depth for dec_tree_opt in clfs]
utils.create_depth_vs_alpha_plot("DT_depth_vs_alpha.png", tree_depths, ccp_alphas)

acc_scores = [accuracy_score(test_data_output, dec_tree_opt.predict(test_data_input)) for dec_tree_opt in clfs]
utils.create_accuracy_vs_alpha_plot("DT_accuracy_vs_alpha.png", acc_scores, ccp_alphas)

# optimizing using different parameters
params = {"min_weight_fraction_leaf": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
          "min_samples_split": range(2, 20, 1),
          "max_depth": range(1, 30)}

for param, param_values in params.items():
    accuracies = utils.get_parametrized_decision_tree_accuracies(
        param, param_values,
        train_data_input, train_data_output,
        test_data_input, test_data_output
    )

    data = pd.DataFrame({'acc_gini': pd.Series(accuracies["acc_gini"]),
                         'acc_entropy': pd.Series(accuracies["acc_entropy"]),
                         param: pd.Series(param_values)})

    utils.create_param_accuracy_plot(f"DT_{param}_acc.png",
                                     param,
                                     data)
'''

# Scale the inputs
scaler = StandardScaler()
pca = decomposition.PCA()
dec_tree = tree.DecisionTreeClassifier()

path = dec_tree.cost_complexity_pruning_path(train_data_input, train_data_output)

pipe = Pipeline(steps=[('std_slc', scaler), ('pca', pca), ('dec_tree', dec_tree)])
n_components = list(range(1,train_data_input.shape[1]+1,1))

criterion = ['gini', 'entropy']
max_depth = [16, 17, 18, 19, 20]

# Hypertuning of model parameters
parameters = dict(pca__n_components = n_components, dec_tree__criterion = criterion, dec_tree__max_depth = max_depth)
clf_gscv = GridSearchCV(pipe, parameters)
clf_gscv.fit(train_data_input, train_data_output)

print('Best Criterion:', clf_gscv.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_gscv.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_gscv.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_gscv.best_estimator_.get_params()['dec_tree'])



clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    clf.fit(train_data_input, train_data_output)
    clfs.append(clf)

print('Number of nodes in the last tree is: {} with cost complexiti $\\alpha$: {}'.format(clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()



train_scores = [clf.score(train_data_input, train_data_output) for clf in clfs]
test_scores = [clf.score(test_data_input, test_data_input) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()
'''

print("Done")
