
def simple_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += expected - real
    return error / len(realOutputs)


def cuadratic_error(expectedOutputs, realOutputs):
    error = 0
    for real, expected in zip(realOutputs, expectedOutputs):
        error += (expected - real)**2
    return error / len(realOutputs)

