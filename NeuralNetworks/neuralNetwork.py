import csv
import random

class NeuralNetwork:
    def __init__(self):
        pass

class NeuralNode:
    def __init__(self):
        pass

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
    traindata = read_csv_line_prepend_bias("./bank-note/train.csv", [0,1,2,3,4])
    testdata  = read_csv_line_prepend_bias("./bank-note/test.csv", [0,1,2,3,4])

    print("RUNNING Stochastic Gradient Descent NN...")

    print("RUNNING Stochastic Gradient Descent NN...")

if __name__ == '__main__':
    main()