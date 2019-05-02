
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
The MNIST dataset of handwritten digits is used for this assignment. You can read more about it [here](http://yann.lecun.com/exdb/mnist/). We've provided a data loader for you in `mnist.py`, but you must download the dataset for yourself. We've provided a data loader for you in `mnist.py`, but you must download and extract the dataset for yourself. Make sure you download all four files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, and `t10k-labels-idx1-ubyte.gz`). Instructions for extracting `.gz` files can be found for Windows and Mac [here](https://www.wikihow.com/Extract-a-Gz-File), and Unix instructions are [here](https://askubuntu.com/questions/25347/what-command-do-i-need-to-unzip-extract-a-tar-gz-file). Do not push these files to your github repository. You will need to use the data loader for some questions below. 

# Coding (1.0 point...but homework turned in without code will get a 0)
There is only one autograded test for this assignment and that is the `test_netid.py` test. You will not have to write any other code to pass the tests. You will, however, still have coding to do for the experiments. 

You must hand in whatever code you did for data loading, your visualizations, and experiments by pushing to github (as you did for all previous assignments). Your code should be in the `code/` directory. If there is code in there, you get the point. 

**NOTE: if we have any doubts about your experiments we reserve the right to check this code to see if your results could have been generated using this code. If we don't believe it, or if there is no code at all, then you may receive a 0 for any free-response answer that would have depended on running code.**


You should make a conda environment for this homework just like you did for previous homeworks. We have included a `requirements.txt`.

# Free-response questions (10 points, total)

#### Understanding SVMs (1 point)
1. (0.5 points) Explain why a support vector machine using a kernel, once trained, does not directly use the decision boundary to classify points. 

2. (0.5 points) If the support vector machine does not directly use the decision boundary to classify points, how does it, in fact, classify points. *Hint, what are the support vectors?*

#### the MNIST data (1 point)
3. (0.5 points) How many images are there in the MNIST data? How many images are there of each digit? How many different people's handwriting? Are the digit images all the same size and orientation? What is the color palette of MNIST (grayscale, black & white, RGB)?

4. (0.5 points) Select one of the digits from the MNIST data. Look through the variants of this digit that different people produced. Show us 3 examples of that digit you think might be challenging for a classifier to correctly classify. Explain why you think they might be challenging.

#### Estimating training time (1.5 points)
5. (0.5 point) Before running any serious experiments, first figure out how long your computer takes to train support vector machines on the MNIST data  Pick one of the **C** values. Try training a linear SVM on 1000 examples from the training set. Train another SVM with 2000 examples, then 4000 examples. Do the same three experiments with a polynomial, and radial basis function (RBF) kernel. Report what **C** value you picked and the time it took to train each of your SVMs in a table with 3 rows (1 kernel per row) and 3 columns (for the size of the training set). Rows and columns should be clearly labeled. (_HINT:_ Use python's built-in `time` module to time your experiments!) 

6. (0.5 points) Given your data from the previous question (and what you know about SVMs from lecture and reading), write a formula to estimate in clock time how long it would take to train an SVM on your machine, as a function of the number of training examples, given each of the 3 kernels. 

#### Selecting training and testing data  (1 point)

7. (0.5 points) Given your formula from the previous question, what size of training set would guarantee a single trial for your SVM takes no longer than 2 minutes? Assume this will determine the size of your training set. Now that you have this, tell us how big your testing set will be.

8. (0.5 points) Now you have to decide how to make a draw from the data that has good coverage. Think about the goals of training and testing sets - we pick good training sets so our classifier generalizes to unseen data and we pick good testing sets to see whether our classifier generalizes. Explain how you should select training and testing sets. (Entirely randomly? Train on digits 0-4, test on 5-9? Train on one group of hand-writers, test on another?). Justify your method for selecting the training and testing sets in terms of these goals. 

#### Finding the best hyperparameters (4 points)
We want to find the best kernel and slack cost, **C**, for handwritten digit recognition on MNIST using a support vector machine. To do this, we're going to try different kernels from the set {Linear, Polynomial, Radial Basis Function}. We will combine each kernel with a variety of **C** values drawn from the set { 0.1, 1, 10 }. This results in 9 variants of the SVM. For each variant (a.k.a. condition) run 20 trials. 

In one trial *trial* you select testing and traing data using your approach from an earlier question. You then select the kernel and **C**. You then train the SVM on the training data until it converges. You then test the trained SVM on the testing data. For this assignment, we'll be using classfication error on the testing data as the outcome of a trail.  Save this data. We'll ask you to show it to us in different ways.

###### Note: You will have to do 180 trials. If each trial takes 2 minutes, you will need to dedicate 6 hours to these experiments. 

###### Note: There is a tutorial about running python code in parallel included in this repo. Though it is not required, it will make running your experiments much quicker! Look for it here: `code/parallel_tutorial.py`.

9. (0.5 point) Create a table with 3 rows (1 kernel per row) and 3 columns (the 3 slack settings). Rows and columns should be clearly labeled. For each condition (combination of slack and kernel), show the following 3 values: the testing error measure **e**, the standard deviation of the error **std** and the number of trials **n**, written in the format: **e(std),n**. 

10. (0.5 points) Make a boxplot graph that plots testing error (vertical) as a function of the slack **C** (horizontal). Use average results across all kernels. Indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

11. (0.5 points) What statistical test should you use to do comparisons between the values of **C** plotted in the previous question? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

12. (0.5 points) What is the result of your statistical test? Is the difference between the best and second best value of C statistically significant? Is this a good or bad value? How did you determine that?

13. (0.5 points) Make a boxplot graph that plots error (vertical) as a function of kernel choice. Average results across all values for C. Don't forget to indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

14. (0.5 points) What statistical test should you use to determine whether the difference between the best and second best kernel is statistically significant? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer.

15. (0.5 points) What is the result of your statistical test? Is the difference between the best and second best value of **C** statistically significant?

16. (0.5 points) Is the combination of kernel and **C** that shows the best error in the table from the previous question the same combination that resulted from considering **C** and kernel independently? Which one do you believe?

#### Putting these results in context (0.5 point)

17. (0.5 points) Compare your results with the [previous results for SVMs found on MNIST](http://yann.lecun.com/exdb/mnist/). What is the best kernel reported there? How does your best kernel do compared to that one?  *Aside: A Gaussian Kernel is a Radial Basis Function kernel* 

#### Showing us your data. 
18. (0.5 points) Put the error from every individual trial into a single table, where the columns are labeled: error, **C** ,  kernel. Each row will list the error rate (on a scale of 0 to 1) for one trial, the value of **C** for that trial and the kernel for that trial.  