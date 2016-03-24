---
title       : Improving your predictions through Random Forests 
description : "What techniques can you use to improve your predictions even more? One possible way is by making use of the machine learning method Random Forest. Namely, a forest is just a collection of trees..."
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:2
## What is a Random Forest

A detailed study of Random Forests would take this tutorial a bit too far. However, since it's an often used machine learning technique, a general understanding and an illustration in R won't hurt.

In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

Before starting with the actual analysis, you first need to meet one big condition of Random Forests: no missing values in your data frame. Let's get to work.

--- type:NormalExercise lang:python xp:100 skills:2
## A Random Forest analysis in Python

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Important variables
