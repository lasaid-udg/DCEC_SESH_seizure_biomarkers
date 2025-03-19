import numpy
from typing import Tuple
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


class MlTools:

    def __init__():
        pass

    @classmethod
    def get_train_test_data(cls, group_1: numpy.array, group_2: numpy.array) -> Tuple:
        """
        Generate the target for each class and return the train set
        and test set
        :param group_1: class 1 instances, [samples x features]
        :param group_2: class 2 instances, [samples x features]
        """
        group_1 = group_1.T
        group_2 = group_2.T
        target_1 = [0] * group_1.shape[0]
        target_2 = [1] * group_2.shape[0]
        targets = numpy.array(target_1 + target_2)
        features = numpy.concatenate([group_1, group_2], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    @classmethod
    def train_and_score_naive_bayes(cls, x_train: numpy.array, x_test: numpy.array,
                                    y_train: numpy.array, y_test: numpy.array) -> float:
        """
        Train a Gaussian Naive Bayes and compute the classification accuracy
        """
        learner = GaussianNB()
        learner.fit(x_train, y_train)
        accuracy = learner.score(x_test, y_test)
        return accuracy