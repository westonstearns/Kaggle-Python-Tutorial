---
title       : Improving your predictions through Random Forests 
description : "What techniques can you use to improve your predictions even more? One possible way is by making use of the machine learning method Random Forest. Namely, a forest is just a collection of trees..."
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:2
## A Random Forest analysis in Python
A detailed study of Random Forests would take this tutorial a bit too far. However, since it's an often used machine learning technique, a general understanding and an illustration in Python won't hurt.

In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

Building a random forest in Python looks almost the same as building a decision tree so we can jup right to it. There are two key differeances however. Firstly a different class is used. And second a new arguments is necessary. Also, we need to import the necessary library from scikit-learn.

- Use `RandomForestClassifier()` class intead of the `DecisionTreeClassifier()` class. 
- `n_estimators` needs to be set when using the `RandomForestClassifier()` class. This argument allows you to set the number of trees you wish to plant and average over.

The latest training and testing data are preloaded for you.


*** =instructions
- Import `RandomForestClassifier` from `from sklearn.ensemble`.
- Build an array with features we used for the most recent tree and call it features_forest.
- Build the random forest with `n_estimators` set to `100`.
- Build an array with the features from the test set to make predicitons. Use this array and the model to compute the predictions.

*** =hint
*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk


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
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, ___)
my_forest = forest.fit(features_forest, target)
my_forest.score(features_forest, target)

#Computing Predictions:test_features, pred_forest
test_features = np.array(___).transpose()
pred_forest = 

```

*** =solution
```{python}

#Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = np.array([train.Pclass, train.Age, train.Sex, train.Fare, train.SibSp, train.Parch, train.Embarked]).transpose()

#Building the Forest: my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=100)
my_forest = forest.fit(features_forest, target)
my_forest.score(features_forest, target)

#Computing Predictions:test_features, pred_forest
test_features = np.array([test.Pclass,test.Age,test.Sex, test.Fare, test.SibSp, test.Parch,test.Embarked]).transpose()
pred_forest = my_forest.predict(test_features)

```

*** =sct

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Important variables



*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct
