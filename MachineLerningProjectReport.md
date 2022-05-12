# **Comparison of  Common Machine Learning Algorithms**

Even if Deep Learning is continuosly growing in terms of importance, classical Machine Learning technquis still represent a cornerstone for people who approach computer science for the first time. There are loads of different techniques, some of thema re specifically devoted to tackle given types of problems, others are multi-porpose.
One popular task is the so called classification problem, where **the model's output is a category** with a  semantic meaning. A classification model attempts to draw some conclusion from observed values. Given one or more inputs a classification model will try to predict the value of one or more outcomes.

Different methods can be implemented to tackle this problem. Our focus is a brief comparative study over four different machine learning supervised techniques:
1. Logistic Regression
2. K Nearest Neighbors
3. Decision Trees
4. Multilayer Perceptron

## 1. Logistic Regression

Logistic regression is the right algorithm to start with classification algorithms. Eventhough, the name ‘Regression’ comes up, it is not a regression model, but a classification model. It uses a logistic function to frame binary output model. The output of the logistic regression will be a probability (0≤x≤1), and can be used to predict the binary 0 or 1 as the output ( if x<0.5, output= 0, else output=1).

**Loss function**
We use **cross entropy** as our loss function. The basic logic here is that, whenever my prediction is badly wrong, (eg : y’ =1 & y = 0), cost will be -log(0) which is infinity.

**Advantages**
-   Easy, fast and simple classification method.
-   θ parameters explains the direction and intensity of significance of independent variables over the dependent variable.
-   Can be used for multiclass classifications also.
-   Loss function is always convex.

**Disadvantages**
-   Cannot be applied on non-linear classification problems.
-   Proper selection of features is required.
-   Good signal to noise ratio is expected.
-   Colinearity and outliers tampers the accuracy of LR model.

 **Hyperparameters**
Logistic regression hyperparameters are mainly two: Learning rate(α) and Regularization parameter(λ). Those have to be tuned properly to achieve high accuracy.

## 2. K-Nearest Neighbors

K-nearest neighbors is a non-parametric method used for classification and regression. It is one of the most easy ML technique used. It is a lazy learning model, with local approximation.

**Advantages**
-   Easy and simple machine learning model.
-   Few hyperparameters to tune.

 **Disadvantages**
-   k should be wisely selected.
-   Large computation cost during runtime if sample size is large.
-   Proper scaling should be provided for fair treatment among features.

**Hyperparameters**
KNN mainly involves two hyperparameters, K value & distance function.
-   K value : how many neighbors to participate in the KNN algorithm. k should be tuned based on the validation error.
-   distance function : in our case, we choose the Minkowski distance because it allows us to work in a N-D space.

**Assumptions**
-   There should be clear understanding about the input domain.
-   feasibly moderate sample size (due to space and time constraints).
-   colinearity and outliers should be treated prior to training.

## 3. Decision Trees

Decision tree is a tree based algorithm used to solve regression and classification problems. An inverted tree is framed which is branched off from a homogeneous probability distributed root node, to highly heterogeneous leaf nodes, for deriving the output.

**Algorithm to select conditions**
For (classification and regression trees), we use *Gini index* as the classification metric. This lets us calculate how well the datapoints are mixed together.

**Advantages**
-   No preprocessing needed on data.
-   No assumptions on distribution of data.
-   Handles colinearity efficiently.
-   Decision trees can provide understandable explanation over the prediction.

**Disadvantages**
-   Chances for overfitting the model if we keep on building the tree to achieve high purity. decision tree pruning can be used to solve this issue.
-   Prone to outliers.
-   Tree may grow to be very complex while training complicated datasets.
-   Looses valuable information while handling continuous variables.

**Hyperparameters**
Decision tree includes many hyperparameters and I will list a few among them.

-   **criterion** : which cost function for selecting the next tree node. Mostly used ones are gini/entropy.
-   **max depth :** it is the maximum allowed depth of the decision tree.
-   **minimum samples split :** It is the minimum nodes required to split an internal node.
-   **minimum samples leaf :** minimum samples that are required to be at the leaf node.


## Comparison between models
In this paragraph, some considerations about performances and effectiveness are reported with the aim of undestranting the best working conditions for each model.

Logistic regression has a convex loss function, so it won't hang in a local minima, whereas for example neaural network may. One important thing to consider is that logistic regression outperforms neural network when training data is less and features are large, since neural networks need large training data. Of course there is a strike also for neural networks since they can support non-linear solutions where for example logistic regression can not. 
Talking about time consumption, KNN is comparatively slower than other competitors like logistic regression and decision trees, but it supports non-linear solutions too. One major downgrade is that, KNN can only output the labels. Lukily, KNN requires less data to achieve a sufficient accuracy respect to neural networks, but it needs lot of hyperparameter tuning compared to KNN.
Finally, let's spend some workd about decision trees. In general, they handle colinearity better, but can not derive the significance of features, hence they are better for a categorical evaluation. Respect to KNN, decision tree supports automatic feature interaction, and it is faster due to KNN’s expensive real time execution. Decision trees perform better when there is a large set of categorical values in the training data. In comparison to neural networks, decision trees are better suited when the scenario demands an explanation over the decision, but when there is sufficient training data, neural networks outperfomr drastically decision trees.

### References:
- [Comparative Study on Classic Machine learning Algorithms | by Danny Varghese | Towards Data Science](https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222)
