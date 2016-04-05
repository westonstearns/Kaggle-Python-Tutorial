---
title       : Improving your predictions through Random Forests 
description : "What techniques can you use to improve your predictions even more? One possible way is by making use of the machine learning method Random Forest. Namely, a forest is just a collection of trees..."
attachments :


--- type:NormalExercise lang:python xp:100 skills:2
## A Random Forest analysis in Python
A detailed study of Random Forests would take this tutorial a bit too far. However, since it's an often used machine learning technique, a general understanding and an illustration in Python won't hurt.

In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

Building a random forest in Python looks almost the same as building a decision tree so we can jump right to it. There are two key differences however. Firstly a different class is used. And second a new arguments is necessary. Also, we need to import the necessary library from scikit-learn.

- Use `RandomForestClassifier()` class instead of the `DecisionTreeClassifier()` class. 
- `n_estimators` needs to be set when using the `RandomForestClassifier()` class. This argument allows you to set the number of trees you wish to plant and average over.

The latest training and testing data are preloaded for you.


*** =instructions
- Import `RandomForestClassifier` from `from sklearn.ensemble`.
- Build an array with features we used for the most recent tree and call it features_forest.
- Build the random forest with `n_estimators` set to `100`.
- Build an array with the features from the test set to make predictions. Use this array and the model to compute the predictions.


*** =hint

When computing the predictions you can use the `.predict()` mothod just like you did with decision trees!

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train["Embarked"] = train["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

train["Age"] = train["Age"].fillna(train["Age"].median())

target = np.array(train.Survived).transpose()

features_two = np.array([train.Pclass,train.Age,train.Sex, train.Fare, train.SibSp, train.Parch,train.Embarked]).transpose()
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)
my_tree_two = my_tree_two.fit(features_two, target)


test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test["Embarked"] = test["Embarked"].fillna("S")

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())

test.Fare[152] = test.Fare.median()

```

*** =sample_code
```{python}
#Import the `RandomForestClassifier`
from sklearn.ensemble import ___

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = np.array(___).transpose()

#Building the Forest: my_forest
n_estimators = 
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, ___, random_state = 1)
my_forest = forest.fit(features_forest, target)

#Print the score of the random forest


#Compute predictions and print the length of the prediction vector:test_features, pred_forest
test_features = np.array(___).transpose()
pred_forest = 
print()
```

*** =solution
```{python}

#Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = np.array([train.Pclass, train.Age, train.Sex, train.Fare, train.SibSp, train.Parch, train.Embarked]).transpose()

#Building the Forest: my_forest
n_estimators = 100
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = n_estimators, random_state = 1)
my_forest = forest.fit(features_forest, target)

#Print the score of the random forest
print(my_forest.score(features_forest, target))

#Compute predictions and print the length of the prediction vector:test_features, pred_forest
test_features = np.array([test.Pclass,test.Age,test.Sex, test.Fare, test.SibSp, test.Parch,test.Embarked]).transpose()
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

```

*** =sct

```{python}
test_function("RandomForestClassifier")
test_object("n_estimators")
test_object("features_forest")
test_function("print",1)
test_object("test_features")
test_function("print",2)
```

--- type:NormalExercise lang:python xp:100 skills:2
## Interpreting and Comparing

Remember how we looked at `.feature_importances_` attribute for the decision trees? Well you can request the same attribute from your random forest as well and interpret the relevance of the included variables.
You might also want to compare the models in some quick and easy way. For this we can use the `.score()` method. `.score()` method takes the features data and the target and computes mean accuracy of your model. You can apply this method to both the forest and individual trees. Remember, this measure should be high but not extreme because that would be a sign of overfitting.

For this exercise you have `my_forest` and `my_tree_two` available to you. The features and target arrays are also ready for use.

*** =instructions
- Explore the feature importance for both models
- Compare the mean accuracy score of the two models

*** =hint

Make sure you are applying the commands to `my_forest` and are using correct arguments.

*** =pre_exercise_code
```{python}
import random
random.seed(1)

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())

target = np.array(train.Survived).transpose()

features_two = np.array([train.Pclass,train.Age,train.Sex, train.Fare, train.SibSp, train.Parch,train.Embarked]).transpose()
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

features_forest = np.array([train.Pclass, train.Age, train.Sex, train.Fare, train.SibSp, train.Parch, train.Embarked]).transpose()
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=100, random_state = 1)
my_forest = forest.fit(features_forest, target)

```


*** =sample_code
```{python}
#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print()

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print()
```

*** =solution
```{python}
#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))
```
*** =sct

```{python}

test_function("print", 1)
test_function("print", 2)
test_function("print", 3)
test_function("print", 4)

```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Conclude and Submit

Based on your finding in the previous exercise determine which feature was of most importance, and for which model.
After this final exercise you will be able to submit your random forest model to Kagle! 

*** =hint

By significance we simly mean the magnitude of the values.

*** =pre_exercise_code

```{python}

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())

target = np.array(train.Survived).transpose()

features_two = np.array([train.Pclass,train.Age,train.Sex, train.Fare, train.SibSp, train.Parch,train.Embarked]).transpose()
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)
my_tree_two = my_tree_two.fit(features_two, target)

features_forest = np.array([train.Pclass, train.Age, train.Sex, train.Fare, train.SibSp, train.Parch, train.Embarked]).transpose()
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=100)
my_forest = forest.fit(features_forest, target)

```

*** =instructions
- `The most important feature was "Age", but it was more significant for "my_tree_two"`
- `The most important feature was "Sex", but it was more significant for "my_tree_two"`
- `The most important feature was "Sex", but it was more significant for "my_forest"`
- `The most important feature was "Age", but it was more significant for "my_forest"`

*** =sct

```{python}

test_mc(correct = 2, msgs = ["Try again", "Correct!", "Try again","Try again"])


success_msg("Great! You just created your first random forest. [Download your csv file](https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/my_solution_forest.csv), and submit the created csv to Kaggle to imporove your score even more")

```
