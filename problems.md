
# Important background

## Scikit learn

You will use [scikit-learn](http://scikit-learn.org/stable/index.html), a machine learning library for Python, to answer questions in this homework. 
You should be running the latest stable version of scikit-learn (0.20.3, as of this writing).
If you want an example of how to train and call a classifier from scikit-learn, have a look at the [man page for the support vector machine](http://scikit-learn.org/stable/modules/svm.html#multi-class-classification).
Most classifiers have similarly good documentation and are called in similar ways.
For easy-to-use model selection, cross validation, etc, check out [the documentation on model selection](http://scikit-learn.org/stable/model_selection.html#model-selection)

## SciPy and statistical tests
Here are some helpful statistical tests to determine whether two samples are drawn from the same underlying distribution.

* If you have paired samples and normally distributed data, use this: [paired samples t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)

* If you have independent samples and normally distributed data, use this: [independent samples t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

* If you have paired samples and data that doesn't follow a normal distribution use this: [Wilcoxon signed-rank test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html?highlight=wilcoxon#scipy.stats.wilcoxon)

* If you have independent samples and data that doesn't follow a normal distribution use this:[Mann–Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html?highlight=mannwhitney#scipy.stats.mannwhitneyu)


## The MNIST dataset
The MNIST dataset of handwritten digits is included with this assignment (train-images-idx3-ubyte, train-labels-idx1-ubyte), and you can read more about it [here](http://yann.lecun.com/exdb/mnist/). We've provided a data loader for you in `mnist.py`, but you must download the dataset for yourself. You will need to use the data loader for some questions below. 

# Coding (0.0 points)
There is only one autograded test for this assignment and that is the `test_netid.py` test. You will not have to write any other code to pass the tests. You will, however, still have coding to do for the experiments. You must hand in whatever code you did for data loading, your visualizations, and experiments by pushing to github (as you did for all previous assignments). Your code should be in the `code/` directory. Homeworks turned in without code will get a 0 on this assignment.

You should make a conda environment for this homework just like you did for previous homeworks. We have included a `requirements.txt`.

# Free-response questions (10 points, total)

#### Understanding SVMs (1.5 points)

1. (0.5 point) Support vector machines use a kernel. People build kernels for things that don’t start out as vectors, such as text documents. Research on the web to find the name of a kernel used on words or strings. Tell us what it is called. Briefly explain how it works *in your own words*, using a toy example. Also give a citation for a research paper that describes it. Include a web link.

2. (0.5 points) Explain why a support vector machine using a kernel, once trained, does not directly use the decision boundary to classify points. 

3. (0.5 points) If the support vector machine does not directly use the decision boundary to classify points, how does it, in fact, classify points. *Hint, what are the support vectors?*


#### the MNIST data (1 point)
8. (0.5 points) How many images are there in the MNIST data? How many images are there of each digit? How many different people's handwriting? Are the digit images all the same size and orientation? What is the color palette of MNIST (grayscale, black & white, RGB)?

9. (0.5 points) Select one of the digits from the MNIST data. Look through the variants of this digit that different people produced. Show us 3 examples that you think might be challenging for a classifier to correctly classify. Explain why you think they might be challenging.

#### Designing the experiment (1.5 points)
We want to find the best kernel and slack cost, **C**, for handwritten digit recognition on MNIST using a support vector machine. To do this, we're going to try different kernels from the set {Linear, Polynomial, Radial Basis Function}. We will combine each kernel with a variety of **C** values drawn from the set { 0.1, 10^0, 10 }. This results in 9 variants of the SVM. We will now design an experiment to determine the best variant.

A *data split* specifies what portion of the data is used for training vs testing. 

Define a *draw* from the data as one random selection of testing/training, given a data split.

A *condition* is a choice of experimental parameters (model parameters). In the case of our SVM experiments, this is a selection of kernel + slack cost.

Call a *trial* one test/train of a model in a condition, given a draw from the data.

10. (0.5 points) Pick a **C** value. Try training a linear SVM on 1000 examples from the training set. Train another SVM with 2000 examples, then 4000 examples. Do the same three experiments with a polynomial, and radial basis function (RBF) filter. Report what **C** value you picked and the time it took to train each of your SVMs. (_HINT:_ Use python's built-in `time` module to time your experiments!) What happened as you added more data? What happened as you changed kernels? What does this tell you about trying to train on the whole training set? Does it seem feasible to do before this homework assignment is due?

11. (0.5 points) Given that you will have to create a subset of the training and testing data, describe a way to pick a subset such that we can trust the results of our SVM. Think about the goals of training and testing sets - we pick good training sets so our classifier generalizes to unseen data and we pick good testing sets to see whether our classifier generalizes. Explain how you should select training and testing sets. (Entirely randomly? Train on digits 0-4, test on 5-9? Train on one group of hand-writers, test on another?). Justify your method for selecting the training and testing sets in terms of these goals. 

12. (0.25 points) How many examples will be in your training set? Testing set? MNIST is already separated into training and testing sets. From which set will you make your training set? Testing set? How many trials per condition will you run? (Keep in mind that we will be asking you to do statistical tests about your SVM results) Why that many?

14. (0.25 points) What evaluation measure will you use to compare the effectiveness of handwritten digit recognition?


###### Note: There is a tutorial about running python code in parallel included in this repo. Use what you've learned there in your experiments; it will make things much quicker for you! Look for it here: `code/parallel_tutorial.py`.

#### Reporting the results (4 points)
15. (0.5 points) Create a table with 3 rows (1 kernel per row) and 3 columns (the 3 slack settings). Rows and columns should be clearly labeled. For each condition (combination of slack and kernel), show the following 3 values: the error measure **e**, the standard deviation of the error **std** and the number of trials **n**, written in the format: **e(std),n**. 
**WARNING: Training each trial with all 9 kernel and slack settings will take a long time! Plan accordingly.**

16. (0.5 points) Make a boxplot graph that plots testing error (vertical) as a function of the slack **C** (horizontal). Use average results across all kernels. Indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

17. (0.5 points) What statistical test should you use to do comparisons between the values of **C** plotted in the previous question? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

18. (0.5 points) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant? Is this a good or bad value? How did you determine that?

19. (0.5 points) Make a boxplot graph that plots error (vertical) as a function of kernel choice. Average results across all values for C. Don't forget to indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

20. (0.5 points) What statistical test should you use to determine whether the difference between the best and second best kernel is statistically significant? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

21. (0.5 points) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant?

22. (0.5 points) Is the combination of kernel and C that shows the best error in the table from the previous question the same combination that resulted from considering C and kernel independently?

#### Putting these results in context (0.5 point)

23. (0.5 points) Compare your results with the [previous results for SVMs found on MNIST](http://yann.lecun.com/exdb/mnist/). What is the best kernel reported there? How does your best kernel do compared to that one?  *Aside: A Gaussian Kernel is a Radial Basis Function kernel* 

