
# Important background

## Scikit learn

You will use [scikit-learn](http://scikit-learn.org/stable/index.html), a machine learning library for Python, to answer questions in this homework. 
You should be running the latest stable version of scikit-learn (0.20.3, as of this writing).
If you want an example of how to train and call a classifier from scikit-learn, have a look at the [man page for the support vector machine](http://scikit-learn.org/stable/modules/svm.html#multi-class-classification).
Most classifiers have similarly good documentation and are called in similar ways.
For easy-to-use model selection, cross validation, etc, check out [the documentation on model selection](http://scikit-learn.org/stable/model_selection.html#model-selection)

## The MNIST dataset
The MNIST dataset of handwritten digits is included with this assignment (train-images-idx3-ubyte, train-labels-idx1-ubyte), and you can read more about it [here](http://yann.lecun.com/exdb/mnist/)
We've provided a data loader for you in `mnist.py`. Here's an example of how you'd visualize a single handwritten digit from MNIST 

```
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
images, labels = load_mnist(digits=[9], path='.')
#Displaying the mean image for digit 9.
plt.imshow(images.mean(axis=0), cmap = 'gray')
plt.show()
```
# Coding (5 points)
Describe coding part.

# Free-response questions (5 points)

#### Understanding SVMs

1. (0.5 point): Support vector machines use a kernel. We don’t need to have the inputs to the kernel be vectors of numbers. They could be anything, as long as a function K exists that calculates the distance between them and the function satisfies certain conditions (e.g. positive-definite). This led people to find kernel for things that don’t start out as vectors. For example, text documents. Research on the web to find the name of a kernel used on words or strings. Tell us what it is called. Briefly explain how it works *in your own words*. Also give a citation for a research paper that describes it. Include a web link.

2. (0.5 points): Explain how a support vector machine is related to a K nearest neighbor classifier. *Hint, think about the support vectors.*

3. (0.5 points): Explain why a support vector machine, once trained, does not directly use the decision boundary to classify points.

#### Looking at the MNIST DATA
4. (0.25 points) How many images are there in the MNIST data? How many images are there of each digit?

5. (0.5 points) Look at 50 examples of one of the digits from the MNIST data. Show us some of the cases that you think might be challenging to be recognized by a classifier. Explain why you think the digits you illustrated in the previous question may be challenging.

6. (0.5 points) You're going to want to do repeatable experiments on MNIST Explain how you should select training and testing sets. Think about the goals of training and testing sets - we pick good training sets so our classifier generalizes to unseen data and we pick good testing sets to see whether our classifier generalizes. Justify your method for selecting the training and testing sets in terms of these goals.

7. (0.5 points) Some question about what is the appropriate statistical test to use, given the way they selected training\testing sets.

#### We want to find the best kernel and slack cost, C, for handwritten digit recognition on MNIST using a support vector machine. Draw kernels from the set {Linear, Polynomial, Radial Basis Function}. Draw C from the set { 10^-2, 10^-1, 10^0, 10, 10^2}. Measure testing error on each combination of kernel and C. For each unique combination of kernel and C repeat this measurement **HOW MANY?** times, each with a different test-train split. Then answer the following questions.

8. (0.25 points) Make a boxplot graph that plots testing error (vertical) as a function of C (horizontal). Use average results across all kernels. 

9. (0.25) What statistical test should you use to determine whether the difference between the best and second best values of C is statistically significant. Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

10. (0.25) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant?

11. (0.25 points) Make a boxplot graph that plots error (vertical) as a function of kernel choice. Average results across all values for C.

12. (0.25) What statistical test should you use to determine whether the difference between the best and second best kernel is statistically significant. Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

13. (0.25) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant?

14. (0.25) Now that you've selected the best value of C and the best kernel, create a table that shows the error for each of the 15 combinations of kernel and C that you tested. 

15. (0.25) Is the combination of kernel and C that shows the best error in the table from the previous question the same combination that resulted from considering C and kernel independently?











# These are some prior year's questions I'm using as inspiration.

(1 point) Select a statistical test to determine whether the sample error means of your
spell-corrector with the two distance measures (standard Levenshtein and the one with
hill-climber learned weights) are significantly different at the 95% confidence level.
Explain why you chose that test. Apply that test to the data from step (B). Report the
results. If, for some reason, you feel the tests discussed in class are not applicable,
consider using the sign test (http://en.wikipedia.org/wiki/Sign_test ) to determine whether
the difference between the sample error medians is significant.
