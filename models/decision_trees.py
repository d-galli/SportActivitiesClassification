import pandas
import time
import pydotplus 
from sklearn import tree, decomposition
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


DIGITS = 5
ACTIVITIES = ["sitting", "standing", "lying on back", "laying on right side","ascending stairs", "descending stairs", "standing in an elevator still",
              "moving around in an elevator", "walking in a parking lot", "walking on a treadmill with a speed of 4 km/h on flat",
              "walking on a treadmill with a speed of 4 km/h on a 15 deg inclined ositions", "running on a treadmill with a speed of 8 km/h",
              "exercising on a stepper", "exercising on a cross trainer", "cycling on an exercise bike in horizontal position",
              "cycling on an exercise bike in vertical positions", "rowing", "jumping", "playing basketball"]

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
correct = 0
total = 0
for i in range(len(predictions)):
    if predictions[i] == test_data_output[i]:
        correct += 1
    total += 1
accuracy = correct / total

print("Test is done. \nThe accuracy for this model is: ", round(accuracy, DIGITS))
print("The overall time reqired by this model is: ", round(test_time - start_time, DIGITS), "seconds")

dot_data = tree.export_graphviz(model, feature_names = train_data_input.columns, class_names = ACTIVITIES, filled = True, out_file = None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("decision_tree.pdf")

path = model.cost_complexity_pruning_path(train_data_input, train_data_output)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker = 'o', drawstyle = "steps-post")
ax.set_xlabel('Effective $\\alpha $')
ax.set_ylabel("Total impurity of leaves")
ax.set_title('Total impurity VS Effective $\\alpha $')

plt.show()

# Scale the inputs
scaler = StandardScaler()
pca = decomposition.PCA()
dec_tree = tree.DecisionTreeClassifier()

pipe = Pipeline(steps=[('std_slc', scaler), ('pca', pca), ('dec_tree', dec_tree)])
n_components = list(range(1,train_data_input.shape[1]+1,1))

criterion = ['gini', 'entropy']
max_depth = [12, 14, 16, 18, 19, 20]

# Hypertuning of model parameters
parameters = dict(pca__n_components = n_components, dec_tree__criterion = criterion, dec_tree__max_depth = max_depth)
print("Optimising the parameters ...")
clf_gscv = GridSearchCV(pipe, parameters)
clf_gscv.fit(train_data_input, train_data_output)

print('Best Criterion:', clf_gscv.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_gscv.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_gscv.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_gscv.best_estimator_.get_params()['dec_tree'])



clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
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
