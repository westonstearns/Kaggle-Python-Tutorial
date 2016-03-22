---
title       : Getting Started with Python
description : In this chapter we will go trough the essential steps that you will need to take before beginning to beuil predictive models.
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise xp:100 skills:2
## How it works
Welcome to our Kaggle Machine Learning Tutorial. In this tutorial you will explore how to tackle Kaggle's Titanic competition using Python and Machine Learning. In case you're new to Python, it's recommended that you first take our free [Introduction to Python for Data Science Tutorial](https://www.datacamp.com/courses/intro-to-python-for-data-science). Furthermore, while not required, familiarity with machine learning techniques is a plus so you can get the maximum out of this tutorial.

In the editor on the right you should type Python code to solve the exercises. When you hit the 'Submit Answer' button, every line of code is interpreted and executed by Python and you get a message whether or not your code was correct. The output of your Python code is shown in the console in the lower right corner. Python makes use of the # sign to add comments; these lines are not run as Python code, so they will not influence your result.

You can also execute Python commands straight in the console. This is a good way to experiment with Python code, as your submission is not checked for correctness.

*** =instructions
- In the editor to the right you see some Python code and annotations. This is what a typical exercise with look like.
- To complete the exercise and see how the interavtive environment works  add the code to compute y and hit the 'Submit Answer' button. Don't forget to print the result.

*** =hint

Just add a line of Python code that calculates the product of 6 and 9, just like the example in the sample code!

*** =pre_exercise_code
```{python}
x = 0
y = 0
```

*** =sample_code
```{python}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
```

*** =solution
```{python}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
y = 6*9
print(y)
```

*** =sct
```{python}
success_msg("Awesome! See how the console shows the result of the Python code you submitted? Now that you're familiar with the interface, let's get down to business!")
```

--- type:NormalExercise xp:100 skills:2
## Get the Data with Pandas
When the Titanic sank, 1502 of the 2224 passengers and crew got killed. One of the main reasons for this high level of casualties was the lack of lifeboats on this self-proclaimed "unsinkable" ship.

Those that have seen the movie know that some individuals were more likely to survive the sinking (lucky Rose) than others (poor Jack). In this course you will learn how to apply machine learning techniques to predict a passenger's chance of surviving using Python.

Let's start with loading in the training and testing set into your Python environment. You will use the training set to build your model, and the test set to validate it. The data is stored on the web as `csv` files; their URLs are already available as character strings in the sample code. You can load this data with the `read_csv()` method from the Pandas library.

*** =instructions
- First import the Pandas library as np.
- Load the test data similarly to how the train data is loaded.
- Print the train DataFrame

*** =hint

*** =pre_exercise_code


*** =sample_code
```{python}
# Import the Pandas library

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

#Print the train DataFrame

```
*** =solution
```{python}
# Import the Pandas library
import pandas as pd
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = train = pd.read_csv(test_url)

#Print the train DataFrame
print(train)
```

*** =sct
```{python}
success_msg("Fantastic. Now you have the data to work with and build your predictive models!")
```


--- type:MultipleChoiceExercise xp:50 skills:2
## Understanding your data

Before starting with the actual analysis, it's important to understand the structure of your data. Both `test` and `train` are DataFrame objects, the ay pandas in Python represents datasets. You can easily explore a data using the `.describe()` method. `.describe()` summurizes the columns/features of the DataFrame, including the count of observations, mean, max ans so on. Another useful trick is to look at the dimentions of the DataFrame. This is done by requesting the `.shape` attribute of your DataFrame object. (ex. `your_data.shape`)

The training and test set are already available in the workspace, as `train` and `test`. Apply `.describe()` method and print the `.shape` attribute of the training set. Which of the following statements is correct?

*** =instructions
- The training set has 891 observations and 12 variables, count for Age is 714.
- The training set has 418 observations and 11 variables, count for Age is 891.
- The testing set has 891 observations and 11 variables, count for Age is 891.
- The testing set has 418 observations and 12 variables, count for Age is 714.

*** =hint


*** =pre_exercise_code
```{python}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sct
```{python}

```

--- type:NormalExercise xp:100 skills:1
## Rose vs Jack, or Female vs Male

How many people in your training set survived the disaster with the Titanic? To see this, you can use the `value_counts()` method in combination with the `.`-operator to select a single column of a DataFrame:

```
# absolute numbers
train.Survived.value_counts()

# percentages
train.Survived.value_counts(normalize = True)
``` 

If you run these commands in the console, you'll see that 549 individuals died (62%) and 342 survived (38%). A simple way prediction heuristic could be: "majority wins". This would mean that you will predict every unseen observation to not survive.

In general, the `table()` command can help you to explore what variables have predictive value. For example, maybe gender could play a role as well? You can explore this using the `table()` function for a two-way comparison on the number of males and females that survived, with this syntax:

```
table(train$Sex, train$Survived)
```

To get proportions, you can again wrap `prop.table()` around `table()`, but you'll have to specify whether you want row-wise or column-wise proportions. This is done by setting the second argument of `prop.table()`, called `margin`, to 1 or 2, respectively.
*** =instructions
*** =hint
*** =pre_exercise_code
*** =sample_code
*** =solution
*** =sct


