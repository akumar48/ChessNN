from random import uniform
from math import tanh
from sys import argv, exit
from time import time
from statistics import mean

#TODO:
- send blaise the code to run. give him 250 epochs of 10 hidden_layers
- for wandb, tell chatgpt to use the test_network function as a loss function for the epoch and log it every epoch with other relevant stats.
- do leaky ReLU homework:
    if leaky works well for negatives:
        - use leaky ReLU in hidden_layers, maybe input/output layers too?
    else:
        - change the weighting system to use values betweeen 0 and 1 in fen2input.py
- increase input layer size to increase information 
- use wandb to tell which inputs are important


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = uniform(-1, 1)
        
    def activate(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.tanh(weighted_sum)

    def tanh(self, x):
        return tanh(x)

class MLP:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs, learning_rate):
        self.input_layer = [Neuron(num_inputs) for _ in range(num_hidden_neurons)]
        self.hidden_layers = [[Neuron(num_hidden_neurons) for _ in range(num_hidden_neurons)] for _ in range(num_hidden_layers)]
        self.output_layer = [Neuron(num_hidden_neurons) for _ in range(num_outputs)]
        self.learning_rate = learning_rate
        
    def forward_propagation(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.input_layer]
        for hidden_layer in self.hidden_layers:
            hidden_outputs = [neuron.activate(hidden_outputs) for neuron in hidden_layer]
        final_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return final_outputs
    
    def train(self, inputs, targets):
        # Perform forward propagation to get the outputs
        hidden_outputs = [neuron.activate(inputs) for neuron in self.input_layer]
        for hidden_layer in self.hidden_layers:
            hidden_outputs = [neuron.activate(hidden_outputs) for neuron in hidden_layer]
        final_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]

        # Calculate output layer errors and update weights and biases
        output_errors = [target - output for target, output in zip(targets, final_outputs)]
        for i in range(len(self.output_layer)):
            neuron = self.output_layer[i]
            for j in range(len(neuron.weights)):
                neuron.weights[j] += self.learning_rate * output_errors[i] * hidden_outputs[j]
            neuron.bias += self.learning_rate * output_errors[i]

        # Backpropagate hidden layer errors and update weights and biases
        for i in reversed(range(len(self.hidden_layers))):
            hidden_layer = self.hidden_layers[i]
            for j in range(len(hidden_layer)):
                error = 0.0
                for neuron in self.output_layer:
                    error += (neuron.weights[j] * output_errors[i])
                neuron = hidden_layer[j]
                for k in range(len(neuron.weights)):
                    neuron.weights[k] += self.learning_rate * error * inputs[k]
                neuron.bias += self.learning_rate * error

    def save_model(self, filename):
        with open(filename, 'w') as file:
            # Write the architecture information
            file.write(f"{len(self.input_layer[0].weights)} {len(self.hidden_layers)} {len(self.hidden_layers[0])} {len(self.output_layer)}\n")
            # Write the weights and biases for each neuron
            for neuron in self.input_layer + [neuron for hidden_layer in self.hidden_layers for neuron in hidden_layer] + self.output_layer:
                file.write(' '.join(map(str, neuron.weights)) + '\n')
                file.write(str(neuron.bias) + '\n')

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'r') as file:
            # Read the architecture information
            num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs = map(int, file.readline().split())
            mlp = cls(num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs)
            # Read the weights and biases for each neuron
            neuron_list = mlp.input_layer + [neuron for hidden_layer in mlp.hidden_layers for neuron in hidden_layer] + mlp.output_layer
            for neuron in neuron_list:
                neuron.weights = list(map(float, file.readline().split()))
                neuron.bias = float(file.readline())
        return mlp

# Update the test functionality to calculate the score per game
def test_network(network, test_data):
    cumulative_score = 0.0
    for test_input, expected_output in test_data:
        output = network.forward_propagation(test_input)
        score_per_game =    mean(output - expected_output)
        cumulative_score += score_per_game
    return cumulative_score

# Load the training and test data
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

    num_inputs = 85
    num_hidden_layers = 5
    num_hidden_neurons = 50
    num_outputs = 37
    learning_rate = .1


    mlp = MLP(num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs, learning_rate)

    # Load the training data
    training_data = read_data(training_data_filename)

    epochs = 100
    for epoch in range(epochs):
        start_time = time()  # Record the start time for each epoch
        print(f"Epoch {epoch + 1}/{epochs}")

        for inputs, targets in training_data:
            mlp.train(inputs, targets)

        elapsed_time = time() - start_time
        print(f"Time elapsed for epoch {epoch + 1}: {elapsed_time:.2f} seconds")

    # Save the trained model to a file
    model_filename = f"nn2_{training_data_filename}_{test_data_filename}.txt"
    mlp.save_model(model_filename)
    print(f"Trained model saved to {model_filename}")

    # Load the test data
    test_data = read_data(test_data_filename)

    # Test the trained MLP
    cumulative_score = test_network(mlp, test_data)
    print("Cumulative Perception Score:", cumulative_score)