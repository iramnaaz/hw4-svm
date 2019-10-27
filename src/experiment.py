'''
This experiment should walk you through a hyperparameter tuning setup.
Althoug you are going to implement your own hyperpramater tuning algorithms
the signature of the functions and composition of classes is pretty similar
to sklearn's way of tuning hyperparameters.
'''

from .grid_search import GridSearchCV
from .mnist import load_mnist
from .circle import load_circle
from .random_search import RandomSearchCV


def run(estimator, search_type, tuned_parameters, dataset, n_iter=5):
    '''
    This function walks you through an hyperparameter tuning example. You can find
    example calls in test_experiment.py in the tests folder.

    1. Load the specified input and target datasets with the given dataloaders.
       The dataset string is either 'circle' or 'mnist'
       You can find the data loaders in mnist.py and circle.py
    2. Initialize a new GridSearch or RandomSearch object depending on the
       search type string (e.g. search type 'grid_search' or 'random_search')
    3. Fit the hyperparameter tuner to the training data
    4. Get the cv_results from the tuner and select the best hyperparameter configuration
       You can select the best parameter configuration by sorting the list of all configs
       by the accuracy of each configuration.
    5. Return the best parameter configuration dictionary for this estimator

    :param estimator: Simply pass in the estimator into your hyperparameter tuner
    :param search_type: Can either be 'grid_search' or 'random_search'
    :param tuned_parameters: Contains a dictionary with parameters to tune and
        the tuning values that should be tested by the tuner.
    :param dataset: Can either be 'mnist' or 'circle' and should load the specified dataset
    :param n_iter: The number of iterations for the Random Search algorithm

    :return: A list of all the configurations your tuner created with the resulting accuracy.
        This list should be sorted by accuracy. The best configuration should be the first
        element of the list.
    '''

    NUMBER_OF_MNIST_SAMPLES = 500

    # Load the right dataset depending on the dataset param
    if dataset == 'mnist':
        inputs, targets = load_mnist()
        inputs = inputs[:NUMBER_OF_MNIST_SAMPLES]
        targets = targets[:NUMBER_OF_MNIST_SAMPLES]
    else:
        inputs, targets = load_circle()

    # Initialize the correct hyperparameter tuner depending on search_type
    if search_type == 'grid_search':
        tuner = GridSearchCV(estimator, tuned_parameters)
    else:
        tuner = RandomSearchCV(estimator, tuned_parameters, n_iter=n_iter)

    # Fit the hyperparameter tuner to the data
    tuner.fit(inputs, targets)

    # Get the results and sort them by accuracy
    results = tuner.cv_results
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Return the configurations of the best experiments
    return results
