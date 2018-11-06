import math

def simple_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += abs(expected - real)
    if len(realOutputs) == 0:
        print("hello")
    return error / len(realOutputs)


def accuracy_net(expectedOutputs, realOutputs):
    accuracy = 0.0
    for real, expected in zip(realOutputs, expectedOutputs):
        if real == expected:
            accuracy += 1
    return accuracy / len(realOutputs)



def cuadratic_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += (expected - real)**2
    return error / len(realOutputs)


def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper