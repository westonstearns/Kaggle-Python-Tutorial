---
title       : Predicting with Trees
description : After making your first predictions in the previous chapter, it's time to bring you to the next level. In chapter 2 you
will be introduced to a fundamental concept in machine learning: decision trees.
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:2
## Intro to decision trees

In the previous chapter you did all the slicing and dicing yourself to find subsets that have a higher chance of surviving. A decision tree automates this process for you, and outputs a classification model, or classifier.

Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat. The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations in that node determine how to predict for new observations that end up in that terminal node.

First let's import the necessary libraries:

*** =instructions
- Import the `numpy` library as `np`
- From `sklearn` import the `tree`


*** =hint

Use the `import` and `as` special keys.

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

```

*** =sample_code
```{python}
#Import the Numpy library

#Import 'tree' from scikit-learn library
from sklearn 

```

*** =solution
```{python}
#Import the Numpy library
import numpy as np

#Import 'tree' from scikit-learn library
from sklearn import tree

```

*** =sct

```{python}
test_import("numpy", same_as = False)
#test_import("sklearn", same_as = False)
```


--- type:NormalExercise lang:python xp:100 skills:2
## Cleaning and Formatting your Data

Before you can begin constructing your trees you need to get your hands dirty and clean the data so that you can use all the features available to you. In the first chapter we saw that the Age variable had some missing value. Missingness is a whole subject with and in itself, but we will use a simple imputation technique where we substitute each missing value with the median of the all present values.

```
train["Age"] = train["Age"].fillna(train["Age"].median())
```

Another problem is that the Sex and Embarked variables are categorical but in a non-numeric format. Thus we will need to assign each class a unique integer so that Python can handle the information. Embarked also has some missing values which you should impute witht the most common class of embarkation, which is `"S"`.


*** =instructions
- Assign the integer 1 to all females
- Impute missing values in `Embarked` with class `S`. Use `.fillna()` method.
- Replace each class of Embarked with a uniques integer. `0` for `S`, `1` for `C`, and `2` for `Q`.
- Print the `Sex` and `Embarked` columns

*** =hint
Use the `.loc` method to select rows, and dont foget the `==` operator.

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

```

*** =sample_code
```{python}
#Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0

#Impute the Embarked variable
train["Embarked"] = 

#Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0

#Print the Sex and Embarked columns

```

*** =solution
```{python}
#Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

#Print the Sex and Embarked columns
print(train.Sex)
print(train.Embarked)
```
*** =sct

```{python}

test_function("print", 1)
test_function("print", 2)

```

--- type:NormalExercise lang:python xp:100 skills:2
## Creating your first decision tree

You will use the `skit-learn` and `numpy` libraries to build your first decision tree. `skit-learn` can be used to create `tree` objects from the `DecisionTreeClassifier` class. The methods that we will require take `numpy` arrays as inputs and therefore we will need to create those from the `DataFrame` that we already have. We will need the following to build a decision tree

- `target`: A one dimensional numpy array containing the target/response from the train data. (Survival in your case)
- `data`: A multidimensional numpy array containing the features/predictors from the train data. (ex. Sex, Age)

To how this would look like, take a look at the sample code below: 

```
target = np.array(train.Survived).transpose()

features = np.array([train.Sex, train.Age]).transpose()

my_tree = tree.DecisionTreeClassifier()

my_tree = my_tree.fit(features, target)

```

One way to quickly see the result of your decision tree is to see the importance of the features that are included. This is done by requesting the `.feature_importances_` attribute of your tree object.

Ok time for you to build your first decision tree in Python! The train and testing data from chapter 1 are available in your workspace.

*** =instructions
- Build a decision tree `my_tree_one` to predict survival based on the variables Passenger Class, Sex, Age, and Passenger Fare.
- Look at the importance of features in your tree.

*** =hint
 To build a tree use the `tree.DecisionTreeClassifier()` syntax.
 You can look at the importance of features using the `.feature_importances_` attribute.



*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

```

*** =sample_code
```{python}
#Print the train data to see the available features



#Create the target and features numpy arrays: target, features_one



#Fit your first decision tree: my_tree_one



#Look at the importance of the included features

```

*** =solution

```{python}
#Print the train data to see the available features
print(train)

#Create the target and features numpy arrays: target, features

target = np.array(train.Survived).transpose()
features_one = np.array([train.Pclass, train.Sex, train.Age, train.Fare]).transpose()

#Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

#Look at the importance of the included features
print(my_tree_one.feature_importances_)

```

*** =sct

```{python}

test_object("target")
test_object("features_one")
test_function("print",2)
#test_object("my_tree_one")

```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Interpreting your decision tree

The `feature_importances_` attribute make it simple to interpret the significance of the predictors you include. Based on your decision tree, what variable plays the most important role to determine whether or not a passenger survived?

*** =instructions
- Passenger Class
- Sex/Gender
- Passenger Fare
- Age

*** =hint
Have a close look at the attribute of your tree. What variable has the greatest coefficient? 

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

target = np.array(train.Survived).transpose()
features_one = np.array([train.Pclass, train.Sex, train.Age,  train.Fare]).transpose()
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)


```

*** =sct

```{python}

test_mc(correct = 2, msgs = ["Try again", "Correct!" "Try again", "Tray again"])


```


--- type:NormalExercise lang:python xp:100 skills:2
## Predict and submit to Kaggle

To send a submission to Kaggle you need to predict the survival rates for the observations in the test set. In the last exercise of the previous chapter we created rather amateuristic predictions based on a single subset or none at all. Luckily, with our decision tree we can make use of some simple functions to "generate" our answer without having to manually perform subsetting.

First you make use of the `.predict()` method. You provide it the model (`my_tree_two`), the values of features from the dataset for which predictions need to be made (`test`). To extract the features we will need to create a numpy array in the same way as we did when training the model. However we need to take care of a small but important problem first. There is a missing value in the Fare feature that needs to be imputed.

Next, you need to make sure your output is in line with the submission requirements of Kaggle: a csv file with exactly 418 entries and two columns: `PassengerId` and `Survived`. So you need to make a new data frame using `DataFrame()`, and create a csv file using `to_csv()` method from Pandas.

*** =instructions
- Impute the missing value for Fare in row 153 with the median of the column.
- Make a prediction on the test set using the `.predict()` method and `my_tree_two`. Assign the result to `my_prediction`.
- Create a data frame `my_solution` containing the solution and the passenger ids from the test set. Make sure the solution is in line with the standards set forth by Kaggle.
- Check the data frame has 418 entries using `.shape` attribute.
- Turn your solution into a csv file with the name `my_solution.csv`, ready for submission. Do not forget to set the `index` argument to `False`.

*** =hint

When making the `test_features` array make sure that you include the same features in the same order as you did with the train data.
To initidate the submition DataFrame simply select the `PassengerId` from the `test` data. Then add the column with predications.


*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

target = np.array(train.Survived).transpose()
features_one = np.array([train.Pclass, train.Sex, train.Age,  train.Fare]).transpose()
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

```

*** =sample_code

```{python}
#Impute the missing value with the median
test.Fare[152] = 

#Extract the features from the test set
test_features = np.array([test.___, ___, ___, ___]).transpose()

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution = test.PassengerId
my_solution['Survive'] = my_prediction

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", ___)
```

*** =solution

```{python}
#Impute the missing value with the median
test.Fare[152] = test.Fare.median()

#Extract the features from the test set
test_features = np.array([test.Pclass, test.Fare, test.SibSp, test.Parch]).transpose()

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution = test.PassengerId
my_solution['Survive'] = my_prediction

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index = False)
```

*** =sct

```{python}

test_object("test_features")
test_object_after_expression("my_prediction")
test_object_after_expression("my_solution")
test_function("pandas.to_csv")
test_function("print")

success_msg("Great! You just created your first decision tree. [Download your csv file](https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/ch2_ex4_solution/my_solution.csv), and submit the created csv to Kaggle to see the result of your effort.")

```

--- type:NormalExercise lang:python xp:100 skills:2
## Overfitting and how to control it.

When you created your first decision tree the default arguments for  `max_depth` and `min_samples_split` were set to `None`. This means that no limit on the depth of your tree was set.  That's a good thing no! Not so fast. We are likely overfitting. This means that while your model describes the training data extremely well, it doesn't generalize to new data, which is frankly the point of prediction. Just look at the Kaggle submission results for the simple model based on Gender and the complex decision tree. Which one does better?

Maybe we can improve the overfit model by making a less complex model? In `DecisionTreeRegressor`, the depth of our model is defined by two parameters:
- the `max_depth` parameter determines when the splitting up of the decision tree stops.
- the `min_samples_split` parameter monitors the amount of observations in a bucket. If a certain threshold is not reached (e.g minimum 10 passengers) no further splitting can be done.

By limiting the complexity of your decision tree you will increase its generality and thus its usefulness for prediction!
*** =instructions
- Include the Siblings/Spouses Abord, Parents/Childeren Aboard, and Embarked features in a new set of features.
- Fit your second tree `my_tree_two` with the new featurs, and control for the the model compelexity by togoling the `max_depth` and `min_samples_split` arguments.


*** =hint

You can always use `train.describe()` in the consol to check the names of the features.

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

target = np.array(train.Survived).transpose()

train["Age"] = train["Age"].fillna(train["Age"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

```

*** =sample_code
```{python}
# Create a new array with the added features: features_two
features_two = np.array([train.Pclass,train.Age,train.Sex, train.Fare, ___, ___, ___]).transpose()

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
my_tree_two = tree.DecisionTreeClassifier(max_depth = ___, min_samples_split = ____)
my_tree_two = 
```


*** =solution
```{python}
# Create a new array with the added features: features_two
features_two = np.array([train.Pclass,train.Age,train.Sex, train.Fare, train.SibSp, train.Parch, train.Embarked]).transpose()

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)
my_tree_two = my_tree_two.fit(features_two, target)
```

*** =sct

```{python}
test_object("features_two")
test_object("my_tree_two")
test_object("max_depth")
test_object("min_samples_split")
```

--- type:NormalExercise lang:python xp:100 skills:2
## Feature-engineering for our Titanic data set

Data Science is an art that benefits from a human element. Enter feature engineering: creatively engineering your own features by combining the different existing variables. 

While feature engineering is a discipline in itself, too broad to be covered here in detail, you will have a look at a simple example by creating your own new predictive attribute: `family_size`.  

A valid assumption is that larger families need more time to get together on a sinking ship, and hence have less chance of surviving. Family size is determined by the variables `SibSp` and `Parch`, which indicate the number of family members a certain passenger is traveling with. So when doing feature engineering, you add a new variable `family_size`, which is the sum of `SibSp` and `Parch` plus one (the observation itself), to the test and train set.

*** =instructions
- Create a new train set `train_two` that differs from `train` only by having an extra column with your feature engineered variable `family_size`.
- Create a new decision tree. Again use the same variables as last time, but add your feature engineered variable `family_size` as well. Save your new model as `my_tree_three`. 

*** =hint

Don't forget to add `1` to when adding column with the new feature.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

target = np.array(train.Survived).transpose()

train["Age"] = train["Age"].fillna(train["Age"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

```

*** =sample_code
```{python}
# create a new train set with the new variable
train_two = train
train_two['family_size'] = 

# Create a new decision tree my_tree_three
features_three = 
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three =

```

*** =solution

```{python}
# create a new train set with the new variable
train_two = train
train_two['family_size'] = train.SibSp + train.Parch + 1

# Create a new decision tree my_tree_three
features_three = np.array([train.Pclass, train.Sex, train.Age, train.Fare, train.SibSp, train.Parch, train.family_size]).transpose()

my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

```

*** =sct

```{python}

test_object("train_two")
test_object("features_three")
test_object("my_tree_three")

```

