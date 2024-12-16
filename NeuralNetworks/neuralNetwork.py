import csv
import random
import math

class NeuralNetwork:
    def __init__(self, values: list, labels: list, epochs: int):
        self.values = values
        self.labels = labels
        self.epochs = epochs
        # self.M = len(values)
        self.N = len(values)
        self.root = NeuralNode(0, 0, [], [])
        self.inputs = []
        self.build_neural_net()
        self.weights = self.stochastic_gradient_descent()
    
    def stochastic_gradient_descent(self):
        '''
        Stochastic Gradient Descent algorithm for Neural Networks.
        '''
        weight = [0] * self.N
        for _ in range(self.epochs):
            pass
        
        return weight
    
    def build_neural_net(self, layers=2):
        '''
        Builds the Neural Network.
        '''
        # Initialize the root node
        for _ in range(self.N):
            self.root.add_child(NeuralNode(0, 0, [self.root], []))
        
        # Initialize the hidden layers
        current = self.root
        for _ in range(layers):
            for _ in range(self.N):
                child = NeuralNode(0, 0, [], [])
                for j in range(1, self.N):
                    child.add_parent(current.children[j])
                    current.children[j].add_child(child)
            current = current.children[len(current.children) // 2]
        self.inputs = current.children

        for i in range(self.N):
            self.inputs[i].value = self.values[i]

    def forward_propagation(self):
        '''
        Forward propagation algorithm for Neural Networks.
        '''
        queue = self.inputs.copy()           
        while queue != []:
            current = queue.pop(0)
            if current.children != []:
                if not current.children[0].visited:
                    queue.insert(0, current.children[0])
                    queue.insert(1, current)
                    continue
            for i in range(len(current.parents)):
                parent = current.parents[i]
                parent.value += current.value * current.parent_weights[i]
                if parent not in queue:
                    queue.append(parent)
            if current.children != []:
                current.value = self.sigmoid(current.value)
            current.visited = True
        
        return self.root.value
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

class NeuralNode:
    """
    A class representing a node in a neural network.

    Parameters:
    -----------
    value : float
        The value of the node.

    d_value : float
        The derivative of the value with respect to some parameter.

    parents : list
        A list of parent nodes that feed into this node.

    children : list
        A list of child nodes that this node feeds into.
        
    Methods:
    --------
    __init__(self, value, d_value, parents, children):
        Initializes a NeuralNode with the given value, derivative value, parents, and children.
    """
    def __init__(self, value, d_value, parents: list[NeuralNetwork], children: list[NeuralNetwork]):
        self.value = value
        self.d_value = d_value
        self.parents = parents
        self.parent_weights = [0] * len(parents)
        self.children = children
        self.children_weights = [0] * len(parents)
        self.visited = False
    
    def add_parent(self, parent, weight=0):
        """
        Adds a parent node to the list of parents.

        Parameters:
        -----------
        parent : NeuralNode
            The parent node to add.
        """
        self.parents.append(parent)
        self.parent_weights.append(weight)
    
    def add_child(self, child, weight=0):
        """
        Adds a child node to the list of children.

        Parameters:
        -----------
        child : NeuralNode
            The child node to add.
        """
        self.children.append(child)
        self.children_weights.append(weight)
    
    def update_parent_weight(self, index, weight):
        """
        Updates the weight of a parent node.

        Parameters:
        -----------
        index : int
            The index of the parent node to update.

        weight : float
            The weight to update the parent node with.
        """
        self.parent_weights[index] = weight
    
    def update_child_weight(self, index, weight):
        """
        Updates the weight of a child node.

        Parameters:
        -----------
        index : int
            The index of the child node to update.

        weight : float
            The weight to update the child node with.
        """
        self.children_weights[index] = weight  

def read_csv_line_prepend_bias(filepath, attributes : list):
    '''
    Creates a values matrix (prepended with 1) and labels vector based off CSV file.
    Assumes final column are labels. 

    Labels are mapped to binary values 1 and -1.

    Parameters
    ----------
    filepath : string
        path to file
    attributes : list
        index of columns in the data set
    
    Returns
    -------
    Tuple containing the values matrix in 0th index and labels vector 1st index
    '''
    values = []
    labels = []
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file, attributes)
        for row in csv_reader:
            tempV = [1]
            for attribute in attributes:
                if attribute < 4:
                    tempV.append(float(row[attribute]))
                else:
                    labels.append(2 * int(row[attribute]) - 1)
            values.append(tempV)

    return (values, labels)

def main():
    # traindata = read_csv_line_prepend_bias("./bank-note/train.csv", [0,1,2,3,4])
    # testdata  = read_csv_line_prepend_bias("./bank-note/test.csv", [0,1,2,3,4])

    # print("RUNNING Stochastic Gradient Descent NN...")

    # print("RUNNING Stochastic Gradient Descent NN...")
    nn = NeuralNetwork([1,1,1], [0], 100)
    nn.forward_propagation()
    print("DONE")
if __name__ == '__main__':
    main()