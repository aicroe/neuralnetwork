import random
from neuralnetwork.NeuralNetwork import NeuralNetwork

def read_data_set(file_name, row_length, desired_output_index, desired_output_map):
    data_set = []
    for line in open(file_name, 'r'):
        values = line.strip().split(',')
        if len(values) == row_length:
            desired_output_key = values[desired_output_index]
            values = values[0:desired_output_index] + values[desired_output_index + 1:]
            data_set.append((list(map(float, values)), desired_output_map[desired_output_key]))
    random.shuffle(data_set)
    return data_set

def train_network(network, train_set, times=1000):
    for _ in range(times):
        for (inputs, desired_output) in train_set:
            network.train(inputs, desired_output)

def threshold(val):
    if val >= 0.5:
        return 1
    else:
        return 0

def success(network_output, desired_output):
    return list(map(threshold, network_output)) == desired_output

def iris_accuracy():
    desired_output_map = {
        'Iris-setosa': [1.0, 0.0, 0.0],
        'Iris-versicolor': [0.0, 1.0, 0.0],
        'Iris-virginica': [0.0, 0.0, 1.0]
    }
    data_set = read_data_set('iris.csv', 5, 4, desired_output_map)
    boundary = int(len(data_set) * 0.5)
    train_set = data_set[0:boundary]
    test_set = data_set[boundary:]

    network = NeuralNetwork(4, [5], 3)
    train_network(network, train_set, 2000)

    success_count = 0
    for (inputs, desired_output) in test_set:
        if success(network.feed(inputs), desired_output):
            success_count = success_count + 1
    return (success_count * 100.0) / len(test_set)

if __name__ == '__main__':
    random.seed(0)
    print('Prediction accuracy: ', iris_accuracy())
