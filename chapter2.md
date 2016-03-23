---
title       : Predicting with Trees
description : After making your first predictions in the previous chapter, it's time to bring you to the next level. In chapter 2 you
will be introduced to a fundamental concept in machine learning: decision trees.
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise xp:100 skills:2
## Intro to decision trees

In the previous chapter you did all the slicing and dicing yourself to find subsets that have a higher chance of surviving. A decision tree automates this process for you, and outputs a flowchart-like structure that is easy to interpret (you'll make one yourself in the next exercise). 

Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat. The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations in that node determine how to predict for new observations that end up in that terminal node.

Before you begin building decision trees, you first need to import the necessary libraries:

*** =instructions
- Import the `numpy` library as `np`
- From `sklearn` import the `tree`
*** =hint
*** =pre_exercise_code
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

--- type:NormalExercise xp:100 skills:2
## Creating your first decision tree

You will use the `skit-learn` and `numpy` libraries to build your first decision tree. `skit-learn` can be used to create `tree` objects from the `DecisionTreeClassifier` class. The methods that we will require take `numpy` arrays as imputs and therefore we will need to create those from the `DataFrame` that we already have. We will need the following to build a decision tree

- `target`: A one dimentional numpy array containing the target/responce from the train data. (Survival in your case)
- `data`: A multidimentional numpy array containing the features/predictors from the train data. (ex. Pclass, Fare)

To how this would look like, take a look at the sample code below: 

```
target = np.array(train.Survived).transpose()

features = np.array([train.Pclass, train.Fare]).transpose()

my_tree = tree.DecisionTreeClassifier()

my_tree = my_tree.fit(features, target)

```

One way to see the result of your decision tree is to see the importance of the features that are included which can be done by requesting the `.feature_importances_` attribute of your tree object.

Ok time for you to build your first decision tree in Python! The train and testing data from chapter 1 are avaliable in your workspace.


*** =instructions
- Build a decision tree `my_tree_two` to predict survival based on the variables Passenger Class, Number of Siblings/Spouses Aboard, Number of Parents/Children Aboard, and Passenger Fare.
- Look at the importance of features in your tree. Did the importance for features in the first tree change?

*** =hint
*** =pre_exercise_code
```{python}
Import pandas an pd
Import numpy as np
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

```
*** =sample_code
```{Python}
#Print the train data to see the avaliable features



#Create the target and features numpy arrays: target, features



#Fit your first decision tree: my_tree_two



#Look at the importance of the included features

```

*** =solution

```{Python}
#Print the train data to see the avaliable features

print(train)

#Create the target and features numpy arrays: target, features

target = np.array(train.Survived).transpose()
features = np.array([train.Pclass, train.Fare, train.SibSp, train.Parch]).transpose()

#Fit your first decision tree: my_tree_two

my_tree_two = tree.DecisionTreeClassifier()
my_tree_two = my_tree.fit(features, target)

#Look at the importance of the included features
my_tree_two.feature_importances_

```

*** =sct

--- type:MultipleChoiceExercise xp:50 skills:2
## Interpreting your decision tree

On the right, you see the decision tree you just created. Looks nice, doesn't it? It's a very clear graph, that is easy to read and to interpret. Also, you see that thanks to the algorithm we can easily take into account more variables as opposed to creating the segments manually. 

Based on your decision tree, what variables play the most important role to determine whether or not a passenger will survive? 

*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct

--- type:NormalExercise xp:100 skills:2
## Predict and submit to Kaggle

To send a submission to Kaggle you need to predict the survival rates for the observations in the test set. In the last exercise of the previous chapter we created rather amateuristic predictions based on a single subset or none at all. Luckily, with our decision tree we can make use of some simple functions to "generate" our answer without having to manually perform subsetting.

First you make use of the rpart `predict()` function. You provide it the model (`my_tree_two`), the dataset for which predictions need to be made (`test`), and the type of prediction (`class`). You can check out the documentation of `predict()` by running `?predict` in the console.

Next, you need to make sure your output is in line with the submission requirements of Kaggle: a csv file with exactly 418 entries and two columns: `PassengerId` and `Survived`. So you need to make a new data frame using `data.frame()`, and create a csv file using `write.csv()`.

*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct

--- type:NormalExercise xp:100 skills:1,6
## Overfitting, the iceberg of decision trees

If you submitted the solution of the previous exercise, you got a result that outperforms a solution using purely gender. Hurray! 

Maybe we can improve even more by making a more complex model? In `rpart`, the depth of our model is defined by two parameters:
- the `cp` parameter determines when the splitting up of the decision tree stops.
- the `minsplit` parameter monitors the amount of observations in a bucket. If a certain threshold is not reached (e.g minimum 10 passengers) no further splitting can be done.

Stated otherwise, if we set `cp` to zero (= no stopping of splits) and minsplit to 2 (= smallest bucket possible) we will create a super model! Or not? You can see the visualization by typing `fancyRpartPlot(super_model)`. Looking complex, right? 

However, if you submit this solution to Kaggle your score will be lower than the score of a simple model based on e.g. gender. Why? Because you went too far when setting the rules for the decisions trees. You created very specific rules based on the data in the training set that are hence only relevant for the training set but that cannot be generalized to unknown sets. You overfitted. So when creating decision trees, always be aware of this danger!

*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct

--- type:NormalExercise xp:100 skills:1,6
## Re-engineering our Titanic data set

Data Science is an art that benefits from a human element. Enter feature engineering: creatively engineering your own features by combining the different existing variables. 

While feature engineering is a discipline in itself, too broad to be covered here in detail, you will have a look at a simple example by creating your own new predictive attribute: `family_size`.  

A valid assumption is that larger families need more time to get together on a sinking ship, and hence have less chance of surviving. Family size is determined by the variables `SibSp` and `Parch`, which indicate the number of family members a certain passenger is traveling with. So when doing feature engineering, you add a new variable `family_size`, which is the sum of `SibSp` and `Parch` plus one (the observation itself), to the test and train set.

*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct

--- type:NormalExercise xp:100 skills:1,6
## Passenger Title and survival rate

Was it coincidence that upper-class Rose survived and third-class passenger Jack not? Let's have a look... 

You have access to a new train and test set named `train_new` and `test_new`. These data sets contain a new column with the name `Title` (referring to Miss, Mr, etc.). `Title` is another example of feature engineering: creating a new variable that possible improves the model.

*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct
