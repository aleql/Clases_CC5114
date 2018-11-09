import math

import numpy as np


def list_sub(list1, list2):
    error = 0.0
    # for l1, l2 in zip(list1, list2):
    #     error += (l1 - l2)
    error = np.linalg.norm(np.array(list1) - np.array(list2))
    return error


def simple_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += list_sub(expected, real)
    return error / len(realOutputs)


def accuracy_net(expectedOutputs, realOutputs):
    accuracy = 0.0
    for real, expected in zip(realOutputs, expectedOutputs):
        pred = prediction_accuracy(expected, real)
        accuracy += pred
        # if real == expected:
        #     accuracy += 1
    return accuracy / len(realOutputs)

def acumulative_error(expected_outputs_list, real_outputs_list, error_type):
    error = 0.0
    for realOutputs, expectedOutputs in zip(real_outputs_list, expected_outputs_list):
        error += error_type(expectedOutputs, realOutputs)
    return error


def prediction_accuracy(expected, real):
    if isinstance(real, (list,)):
        if np.argmax(real) == np.argmax(expected):
            return 1.0
    elif isinstance(real, (int,)):
        if real == expected:
            return 1
    return 0.0
    # return sum(1 for x,y in zip(expected, real) if x == y) / len(real)

def cuadratic_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += list_sub(expected, real)**2
    return error / len(realOutputs)


def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     return list(np.exp(x) / np.sum(np.exp(x), axis=0))


def identity(value):
    return value

# def argmax(values):
#     values = list(softmax(values))
#     return np.argmax(values) + 1

def index_to_list(index, size=3):
    index = index[0]
    return [0 if i != index - 1 else 1 for i in range(size)]

# print(index_to_list(3))
# x = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
# x = list(softmax(x))
# #print(x.index(max(x)) + 1)
# print( np.argmax(x))