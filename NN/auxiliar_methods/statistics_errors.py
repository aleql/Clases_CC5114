
def simple_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += abs(expected - real)
    if len(realOutputs) == 0:
        print("hello")
    return error / len(realOutputs)


def cuadratic_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += (expected - real)**2
    return error / len(realOutputs)

