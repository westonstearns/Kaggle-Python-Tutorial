---
title       : Improving your predictions through Random Forests 
description : "What techniques can you use to improve your predictions even more? One possible way is by making use of the machine learning method Random Forest. Namely, a forest is just a collection of trees..."
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:2
## A Random Forest analysis in Python
A detailed study of Random Forests would take this tutorial a bit too far. However, since it's an often used machine learning technique, a general understanding and an illustration in Python won't hurt.

In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

Building a random forest in Python looks almost the same as building a decision tree so we can jup right to it. There are two key differeances however. Firstly a different class is used. And second a new arguments is necessary.

- Use `RandomForestClassifier()` class intead of the `DecisionTreeClassifier()` class.
- `n_estimators` needs to be set when using the `RandomForestClassifier()` class. This argument allows you to set the number of trees you wish to plant and average over.

*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Important variables



*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct
