
# Important background

## Scikit learn

You will use [scikit-learn](http://scikit-learn.org/stable/index.html), a machine learning library for Python, to answer questions in this homework. 
You should be running the latest stable version of scikit-learn (0.20.3, as of this writing).
If you want an example of how to train and call a classifier from scikit-learn, have a look at the [man page for the support vector machine](http://scikit-learn.org/stable/modules/svm.html#multi-class-classification).
Most classifiers have similarly good documentation and are called in similar ways.
For easy-to-use model selection, cross validation, etc, check out [the documentation on model selection](http://scikit-learn.org/stable/model_selection.html#model-selection)

## SciPy and statistical tests
Here are some helpful statistical tests to determine whether two samples are drawn from the same underyling distribution.

* If you have paired samples and normally distributed data, use this: [paired samples ttest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)

* If you have independent samples and normally distributed data, use this: [independent sampes ttest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

* If you have paired samples and data that doesn't follow a normal distribution use this: [Wilcoxon signed-rank test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html?highlight=wilcoxon#scipy.stats.wilcoxon)

* If you have independent samples and data that doesn't follow a normal distribution use this:[Mann–Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html?highlight=mannwhitney#scipy.stats.mannwhitneyu)


## The MNIST dataset
The MNIST dataset of handwritten digits is included with this assignment (train-images-idx3-ubyte, train-labels-idx1-ubyte), and you can read more about it [here](http://yann.lecun.com/exdb/mnist/). 
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
# Coding (1.5 points???)
Describe coding part.

# Free-response questions (8 points, total)

#### Understanding SVMs (1.5 points)

1. (0.5 point): Support vector machines use a kernel. People build kernels for things that don’t start out as vectors, such as text documents. Research on the web to find the name of a kernel used on words or strings. Tell us what it is called. Briefly explain how it works *in your own words*. Also give a citation for a research paper that describes it. Include a web link.

2. (0.5 points): Explain how a support vector machine is related to a K nearest neighbor classifier. *Hint, think about the support vectors.*

3. (0.5 points): Explain why a support vector machine, once trained, does not directly use the decision boundary to classify points.

#### the MNIST data (1 point)
4. (0.5 points) How many images are there in the MNIST data? How many images are there of each digit? How many different people's handwriting? Are the digit images all the same size and orientation? What is the color pallate of MNIST (grayscale, black & white, RGB)?

5. (0.5 points) Look at 50 examples of one of the digits from the MNIST data. Show us some of the cases that you think might be challenging to be recognized by a classifier. Explain why you think the digits you illustrated in the previous question may be challenging.

#### Designing the experiment (1.5 points)
We want to find the best kernel and slack cost, **C**, for handwritten digit recognition on MNIST using a support vector machine. To do this, we're going to try different kernels from the set {Linear, Polynomial, Radial Basis Function}. We will combine each kernel with a variety of **C** values drawn from the set { 10^-2, 10^-1, 10^0, 10, 10^2}. This results in 15 variants of the SVM. We will now design an experiment to determine the best variant.

A *data split* specifies what portion of the data is used for training vs testing. 

Define a *draw* from the data as one random selection of testing/training, given a data split.

A *condition* is a choice of experimental parameters (model parameters). In the case of our SVM experimentts, this is a selection of kernel + slack cost.

Call a *trail* one test/train of a model in a condition, given a draw from the data.

6. (0.5 points) We want to see how well different varients of SVM can classify the handwritten digits in MNIST. Think about the goals of training and testing sets - we pick good training sets so our classifier generalizes to unseen data and we pick good testing sets to see whether our classifier generalizes. Explain how you should select training and testing sets. (Entirely randomly? Train on digits 0-4, test on 5-9? Train on one group of handwriters, test on another?). Justify your method for selecting the training and testing sets in terms of these goals. 

7. (0.25 points) What will your test train data split be? Why did you pick that?

8. (0.25 points) Given the previous constraints on how to select testing/training, explain how you'll get different random draws from the data so that trials are independent.

9. (0.25 points) How many trials per condition will you run? Why that many?

10. (0.25 points) What evaluation measure will you use to compare the effectiveness of handwritten digit recognition?

#### Reporting the results (4 points)
11. (0.5 points) Create a table with 3 rows (1 kernel per row) and 5 columns (the 5 slack settings). Rows and columns should be clearly labeled. For each condition (combination of slack and kernel), show the following 3 values: the error measure **e**, the standard deviation of the error **std** and the number of trials **n**, written in the format: **e(std),n**. 

12. (0.5 points) Make a boxplot graph that plots testing error (vertical) as a function of the slack **C** (horizontal). Use average results across all kernels. Indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

13. (0.5) What statistical test should you use to do pairwise comparisons between the values of **C** plotted in the previous question? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

14. (0.5) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant?

15. (0.5 points) Make a boxplot graph that plots error (vertical) as a function of kernel choice. Average results across all values for C. Don't forget to indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

15. (0.5) What statistical test should you use to determine whether the difference between the best and second best kernel is statistically significant? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

16. (0.5) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant?

17. (0.5) Is the combination of kernel and C that shows the best error in the table from the previous question the same combination that resulted from considering C and kernel independently?

#### Putting these results in context (0.5 point)

18. (0.5) Compare your results with [previous results found on MNIST](http://yann.lecun.com/exdb/mnist/). What is the best kernel reported there? Do your results agree with theirs? 

