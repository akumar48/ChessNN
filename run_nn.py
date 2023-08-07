import random
import math
import sys
import time

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [0.0] * num_inputs
        self.bias = 0.0

    def activate(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.tanh(weighted_sum)

    def tanh(self, x):
        return math.tanh(x)

class MLP:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs):
        self.input_layer = [Neuron(num_inputs) for _ in range(num_hidden_neurons)]
        self.hidden_layers = [[Neuron(num_hidden_neurons) for _ in range(num_hidden_neurons)] for _ in range(num_hidden_layers)]
        self.output_layer = [Neuron(num_hidden_neurons) for _ in range(num_outputs)]

    def forward_propagation(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.input_layer]
        for hidden_layer in self.hidden_layers:
            hidden_outputs = [neuron.activate(hidden_outputs) for neuron in hidden_layer]
        final_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return final_outputs

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

def process_questions(model, questions_filename):
    with open(questions_filename, "r") as file:
        lines = file.readlines()

    for line in lines:
        input_data = [float(bit) for bit in line.strip().split(",")]
        output = model.forward_propagation(input_data)
        print("Input:", input_data)
        print("Output:", output)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python neural_net_3.py <model_file> <questions_file>")
        sys.exit(1)

    model_filename = sys.argv[1]
    questions_filename = sys.argv[2]

    # Load the model
    model = MLP.load_model(model_filename)

    # Process questions and print outputs
    process_questions(model, questions_filename)
