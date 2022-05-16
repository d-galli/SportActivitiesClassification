import pandas
import time
import pydotplus
import seaborn as sns
import numpy as np
from sklearn import tree, decomposition
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DIGITS = 5
CV_FOLDS = 7
ACTIVITIES = ["sitting", "standing", "lying on back", "laying on right side","ascending stairs", "descending stairs", "standing in an elevator still",
              "moving around in an elevator", "walking in a parking lot", "walking on a treadmill with a speed of 4 km/h on flat",
              "walking on a treadmill with a speed of 4 km/h on a 15 deg inclined ositions", "running on a treadmill with a speed of 8 km/h",
              "exercising on a stepper", "exercising on a cross trainer", "cycling on an exercise bike in horizontal position",
              "cycling on an exercise bike in vertical positions", "rowing", "jumping", "playing basketball"]

HEADERS = ["T_xacc", "T_yacc", "T_zacc", "T_xgyro", "T_ygyro", "T_zgyro", "T_xmag", "T_ymag", "T_zmag",
        "RA_xacc", "RA_yacc", "RA_zacc", "RA_xgyro", "RA_ygyro", "RA_zgyro", "RA_xmag", "RA_ymag", "RA_zmag",
        "LA_xacc", "LA_yacc", "LA_zacc", "LA_xgyro", "LA_ygyro", "LA_zgyro", "LA_xmag", "LA_ymag", "LA_zmag",
        "RL_xacc", "RL_yacc", "RL_zacc", "RL_xgyro", "RL_ygyro", "RL_zgyro", "RL_xmag", "RL_ymag", "RL_zmag",
        "LL_xacc", "LL_yacc", "LL_zacc", "LL_xgyro", "LL_ygyro", "LL_zgyro", "LL_xmag", "LL_ymag", "LL_zmag",
        "var_T_xacc", "var_T_yacc", "var_T_zacc", "var_T_xgyro", "var_T_ygyro", "var_T_zgyro", "var_T_xmag", "var_T_ymag", "var_T_zmag",
        "var_RA_xacc", "var_RA_yacc", "var_RA_zacc", "var_RA_xgyro", "var_RA_ygyro", "var_RA_zgyro", "var_RA_xmag", "var_RA_ymag", "var_RA_zmag",
        "var_LA_xacc", "var_LA_yacc", "var_LA_zacc", "var_LA_xgyro", "var_LA_ygyro", "var_LA_zgyro", "var_LA_xmag", "var_LA_ymag", "var_LA_zmag",
        "var_RL_xacc", "var_RL_yacc", "var_RL_zacc", "var_RL_xgyro", "var_RL_ygyro", "var_RL_zgyro", "var_RL_xmag", "var_RL_ymag", "var_RL_zmag",
        "var_LL_xacc", "var_LL_yacc", "var_LL_zacc", "var_LL_xgyro", "var_LL_ygyro", "var_LL_zgyro", "var_LL_xmag", "var_LL_ymag", "var_LL_zmag"]

print("Importing data ...")

train_data = pandas.read_csv("../sportsDataset/TrainingDataset.csv")
test_data = pandas.read_csv("../sportsDataset/TestDataset.csv")

train_data_input = train_data.loc[:, train_data.columns != 'activity_index']
train_data_output = train_data.loc[:, train_data.columns == 'activity_index'].values.ravel()
test_data_input = test_data.loc[:, test_data.columns != 'activity_index']
test_data_output = test_data.loc[:, test_data.columns == 'activity_index'].values.ravel()

print("Scaling data ...")
scaler = StandardScaler()
scaler.fit(train_data_input)

train_data_input = scaler.transform(train_data_input)
test_data_input = scaler.transform(test_data_input)

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

print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")
print("The classification report is:\n")
print(classification_report(test_data_output, predictions, zero_division = 0))
print("Confusion matrix:")
print(confusion_matrix(test_data_output, predictions))


print("Exporting the tree ...")
new_train_data_input = pandas.DataFrame(train_data_input, columns = HEADERS)
dec_tree_data = tree.export_graphviz(model, feature_names = new_train_data_input.columns, class_names = ACTIVITIES, filled = True, out_file = None) 
graph = pydotplus.graph_from_dot_data(dec_tree_data) 
graph.write_pdf("decision_tree.pdf")
print('Tree saved as PDF')

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


path = model.cost_complexity_pruning_path(train_data_input, train_data_output)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker = '.', drawstyle = "steps-post")
ax.set_xlabel('Effective $\\alpha $', fontsize = 15)
ax.set_ylabel("Total impurity of leaves", fontsize = 15)
ax.set_title('Total impurity VS Effective $\\alpha $', fontsize = 20)

#plt.show()

print("Cross Validation")
dt_cv = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 19)
cv_scores = cross_val_score(dt_cv, train_data_input, train_data_output, cv = CV_FOLDS)
print("Considering ", CV_FOLDS, " randomly created  groups and performing the cross validation, the accuracy values obtained are:\n", cv_scores)
print('which lead to a mean value of: {}'.format(round(np.mean(cv_scores), DIGITS)))

print("Optimising model parameters ...")

clfs = []

for ccp_alpha in ccp_alphas:
    dec_tree_opt = tree.DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    dec_tree_opt.fit(train_data_input, train_data_output)
    clfs.append(dec_tree_opt)

tree_depths = [dec_tree_opt.tree_.max_depth for dec_tree_opt in clfs]
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1], marker = '.')
plt.xlabel("Effective $ \\alpha$", fontsize = 15)
plt.ylabel("Depth of the tree", fontsize = 15)
plt.xscale("log")
plt.title('Total depth  VS Effective $\\alpha $', fontsize = 20)

acc_scores = [accuracy_score(test_data_output, dec_tree_opt.predict(test_data_input)) for dec_tree_opt in clfs]

plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1], marker = '.')
plt.xlabel("Effective $ \\alpha$", fontsize = 15)
plt.ylabel("Accuracy", fontsize = 15)
plt.xscale("log")
plt.title('Accuracy VS Effective $\\alpha $', fontsize = 20)

#plt.show()

fractions = [0,0.1,0.2,0.3,0.4,0.5]
min_weight_fraction_leaf = []
acc_gini = []
acc_entropy = []
for i in fractions:
    dtree = tree.DecisionTreeClassifier(criterion = 'gini', min_weight_fraction_leaf = i )
    dtree.fit(train_data_input, train_data_output)
    pred = dtree.predict(test_data_input)
    acc_gini.append(accuracy_score(test_data_output, pred))
    
    ####
    
    dtree = tree.DecisionTreeClassifier(criterion = 'entropy',min_weight_fraction_leaf = i)
    dtree.fit(train_data_input, train_data_output)
    pred = dtree.predict(test_data_input)
    acc_entropy.append(accuracy_score(test_data_output, pred))
    
    ####
    
    min_weight_fraction_leaf.append(i)
    
d = pandas.DataFrame({'acc_gini':pandas.Series(acc_gini), 'acc_entropy':pandas.Series(acc_entropy), 'min_weight_fraction_leaf':pandas.Series(min_weight_fraction_leaf)})
# visualizing changes in parameters
plt.figure(figsize=(10, 6))
plt.plot('min_weight_fraction_leaf','acc_gini', data=d, label='gini')
plt.plot('min_weight_fraction_leaf','acc_entropy', data=d, label='entropy')
plt.xlabel('min_weight_fraction_leaf')
plt.ylabel('accuracy')
plt.legend()


min_samples_split = []
acc_gini = []
acc_entropy = []
for i in range(2,20,1):
    dtree = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_split = i )
    dtree.fit(train_data_input, train_data_output)
    pred = dtree.predict(test_data_input)
    acc_gini.append(accuracy_score(test_data_output, pred))
    
    ####
    
    dtree = tree.DecisionTreeClassifier(criterion = 'entropy',min_samples_split = i)
    dtree.fit(train_data_input, train_data_output)
    pred = dtree.predict(test_data_input)
    acc_entropy.append(accuracy_score(test_data_output, pred))

    ####
    min_samples_split.append(i)

d = pandas.DataFrame({'acc_gini':pandas.Series(acc_gini), 'acc_entropy':pandas.Series(acc_entropy),  'min_samples_split':pandas.Series(min_samples_split)})
# visualizing changes in parameters
plt.figure(figsize=(10, 6))
plt.plot('min_samples_split','acc_gini', data=d, label='gini')
plt.plot('min_samples_split','acc_entropy', data=d, label='entropy')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.legend()

max_depth = []
acc_gini = []
acc_entropy = []

for i in range(1,30):
    dtree = tree.DecisionTreeClassifier(criterion='gini',max_depth=i )
    dtree.fit(train_data_input, train_data_output)
    pred = dtree.predict(test_data_input)
    acc_gini.append(accuracy_score(test_data_output, pred))

    ####

    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    dtree.fit(train_data_input, train_data_output)
    pred = dtree.predict(test_data_input)
    acc_entropy.append(accuracy_score(test_data_output, pred))

    ####

    max_depth.append(i)

d = pandas.DataFrame({'acc_gini':pandas.Series(acc_gini), 'acc_entropy':pandas.Series(acc_entropy), 'max_depth':pandas.Series(max_depth)})
# visualizing changes in parameters
plt.figure(figsize=(10, 6)) 
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()

plt.show()
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