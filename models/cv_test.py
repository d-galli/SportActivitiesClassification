import os.path
import sys

from sklearn import neighbors, linear_model, tree

from utils import utils

print("Importing data ...")

parent_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
train_data_input, train_data_output = \
    utils.get_splitted_dataset(
        os.path.join(parent_dir, "../sports_dataset/activities_dataset.csv")
    )
print("Data imported")

# Cross Validation

print("Cross Validation")
#classifier_cv = neighbors.KNeighborsClassifier(n_neighbors = 19, metric = 'minkowski', p = 2)
#classifier_cv = linear_model.LogisticRegression(max_iter=8000)
classifier_cv = tree.DecisionTreeClassifier(criterion='entropy', max_depth=19)

cv_scores = utils.get_cross_validation_score(classifier_cv, train_data_input, train_data_output)

print("Done")

