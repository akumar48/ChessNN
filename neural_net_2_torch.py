import random
import math
import sys
import torch
import time

class Neuron:
    def __init__(self, num_inputs):
        self.weights = torch.rand(num_inputs, requires_grad=True)
        self.bias = torch.rand(1, requires_grad=True)
        
    def activate(self, inputs):
        inputs = inputs.clone().detach()
        weighted_sum = torch.dot(self.weights, inputs) + self.bias
        return torch.tanh(weighted_sum)

class MLP:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs):
        self.input_layer = [Neuron(num_inputs) for _ in range(num_hidden_neurons)]
        self.hidden_layers = [[Neuron(num_hidden_neurons) for _ in range(num_hidden_neurons)] for _ in range(num_hidden_layers)]
        self.output_layer = [Neuron(num_hidden_neurons) for _ in range(num_outputs)]  # Use num_outputs here

    def forward_propagation(self, inputs):
        hidden_outputs = torch.tensor(inputs, requires_grad=False)
        for neuron in self.input_layer:
            hidden_outputs = torch.tensor([neuron.activate(hidden_outputs) for neuron in self.input_layer])
        for hidden_layer in self.hidden_layers:
            hidden_outputs = torch.tensor([neuron.activate(hidden_outputs) for neuron in hidden_layer])
        final_outputs = torch.tensor([neuron.activate(hidden_outputs) for neuron in self.output_layer])
        return final_outputs

    def train(self, inputs, targets, learning_rate=0.1):
        inputs = torch.tensor(inputs, requires_grad=False)
        targets = torch.tensor(targets, requires_grad=False)

        hidden_outputs = inputs.clone().detach()
        for neuron in self.input_layer:
            hidden_outputs = torch.tensor([neuron.activate(hidden_outputs) for neuron in self.input_layer])
        for hidden_layer in self.hidden_layers:
            hidden_outputs = torch.tensor([neuron.activate(hidden_outputs) for neuron in hidden_layer])

        final_outputs = torch.tensor([neuron.activate(hidden_outputs) for neuron in self.output_layer])

        with torch.no_grad():
            for i, neuron in enumerate(self.output_layer):
                output_error = targets[i] - final_outputs[i]  # Calculate error for each output neuron (one-dimensional slicing)
                neuron.weights += learning_rate * output_error * hidden_outputs
                neuron.bias += learning_rate * torch.sum(output_error)


    def save_model(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{len(self.input_layer[0].weights)} {len(self.hidden_layers)} {len(self.hidden_layers[0])} {len(self.output_layer)}\n")
            for neuron in self.input_layer + [neuron for hidden_layer in self.hidden_layers for neuron in hidden_layer] + self.output_layer:
                file.write(' '.join(map(str, neuron.weights)) + '\n')
                file.write(str(neuron.bias) + '\n')

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'r') as file:
            num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs = map(int, file.readline().split())
            mlp = cls(num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs)
            for neuron in mlp.input_layer + [neuron for hidden_layer in mlp.hidden_layers for neuron in hidden_layer] + mlp.output_layer:
                neuron.weights = list(map(float, file.readline().split()))
                neuron.bias = float(file.readline())
        return mlp

def test_network(network, test_data):
    cumulative_score = 0.0
    with torch.no_grad():
        for test_input, expected_output in test_data:
            output = network.forward_propagation(test_input)
            score_per_game = output[0] - expected_output[0]
            cumulative_score += score_per_game
    return cumulative_score / len(test_data)

def read_data(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        input_data, output_data = line.strip().split(" ")
        inputs = [float(bit) for bit in input_data.split(",")]
        outputs = [float(bit) for bit in output_data.split(",")]
        data.append((inputs, outputs))

    return data

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python neural_net.py <training_data_file> <test_data_file>")
        sys.exit(1)

    training_data_filename = sys.argv[1]
    test_data_filename = sys.argv[2]

    # Load the training data
    training_data = read_data(training_data_filename)

    # Determine the number of inputs and outputs from the data
    num_inputs = len(training_data[0][0])
    num_outputs = len(training_data[0][1])

    num_hidden_layers = 5
    num_hidden_neurons = 85

    mlp = MLP(num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs)

    epochs = 100
    for epoch in range(epochs):
        start_time = time.time()  # Record the start time for each epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        for inputs, targets in training_data:
            mlp.train(inputs, targets)

        elapsed_time = time.time() - start_time
        print(f"Time elapsed for epoch {epoch + 1}: {elapsed_time:.2f} seconds")

    model_filename = f"nn2t_{training_data_filename}_{test_data_filename}.txt"
    mlp.save_model(model_filename)
    print(f"Trained model saved to {model_filename}")

    # Load the test data
    test_data = read_data(test_data_filename)

    # Determine the number of inputs and outputs for the test data
    num_test_inputs = len(test_data[0][0])
    num_test_outputs = len(test_data[0][1])

    if num_test_inputs != num_inputs or num_test_outputs != num_outputs:
        print("Error: Test data does not match the size of training data.")
        sys.exit(1)

    cumulative_score = test_network(mlp, test_data)
    print("Cumulative Perception Score:", cumulative_score)
