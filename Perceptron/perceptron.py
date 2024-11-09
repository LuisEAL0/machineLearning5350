import csv
import random

class Perceptron:
    ''' Perceptron model to make predictions based on given data '''

    def __init__(self, values: list, labels: list, perceptron_algorithm, epochs, r):
        """
        Perceptron takes in a set of values and labels then creates a weight vector
        that is used for prediction.

        Parameters
        ----------
        values : list
            attributes of a data set (should be preprocessed to include bias term)

        labels : list
            outputs of a data set (must be binary values of -1 and 1)
        
        perceptron_algorithm: function
            variant of perceptron
        
        epochs: int
            number of epochs
        
        r: float
            learning rate
        
        Internals
        ----
        prediction : list
            prediction using learned weight vector
        
        weight : list
            learned weight vector(s) with counts
        
        M : integer
            Number of rows in data set
        
        N : integer
            Number of columns in data set
        """
        self.M = len(values)
        self.N = len(values[0])
        self.prediction = None
        self.weight = perceptron_algorithm(self, values, labels, epochs, r)
    
    def standard(self, values, labels, epochs, r):
        ''' 
        Standard perceptron algorithm

        Returns
        -------
        Learned weight vector
        '''
        weight = [0] * self.N
        for _ in range(epochs):
            shuffled_values, shuffled_labels = self.shuffle(values, labels)
            for row in range(self.M):
                predict = 0
                epoch_weight = weight.copy()
                # Generate prediction and potential new epoch weight
                for column in range(len(values[row])):
                    predict += (shuffled_values[row][column] * weight[column])
                    epoch_weight[column] = weight[column] + (r * shuffled_labels[row] * shuffled_values[row][column])

                # Was prediction incorrect?
                if shuffled_labels[row] * predict <= 0:
                    weight = epoch_weight
        
        self.prediction = self.standard_prediction(values, weight)

        return weight
    
    def standard_prediction(self, values, weight):
        '''
        Creates prediction for the standard perceptron algorithm

        Returns
        -------
        List of predictions
        '''
        prediction = [0] * len(values)
        for i in range(len(values)):
            for j in range(len(values[0])):
                prediction[i] += values[i][j] * weight[j]

            # Encode prediction to binary label
            if prediction[i] > 0:
                prediction[i] = 1
            else:
                prediction[i] = -1

        return prediction

    def shuffle(self, values, labels):
        '''
        Shuffle values and labels congruent to another

        Returns
        -------
        Returns values and labels shuffled

        Reference
        ---------
        https://stackoverflow.com/a/23289591
        '''
        combo = list(zip(values, labels))
        random.shuffle(combo)

        values, labels = zip(*combo)

        return [values, labels]

    def voted(self, values, labels, epochs, r):
        ''' 
        Voted perceptron algorithm

        Returns
        -------
        Family of learned weight vector
        '''
        weight = [0] * self.N
        weight_family = []
        for _ in range(epochs):
            correct_m = 0
            for row in range(self.M):
                predict = 0
                epoch_weight = weight.copy()
                # Generate prediction and potential new epoch weight
                for column in range(len(values[row])):
                    predict += (values[row][column] * weight[column])
                    epoch_weight[column] = weight[column] + (r * labels[row] * values[row][column])

                # Was prediction incorrect?
                if labels[row] * predict <= 0:
                    weight_family.append((weight.copy(), correct_m))
                    weight = epoch_weight
                    correct_m = 1
                else:
                    correct_m += 1

        self.prediction = self.voted_prediction(values, weight_family)

        return weight_family
    
    def voted_prediction(self, values, weights):
        '''
        Creates prediction for the voted perceptron algorithm

        Returns
        -------
        List of predictions
        '''
        predictions = [0] * len(values)
        votes = [0] * len(values)

        for weight, correct in weights:
            prediction = self.standard_prediction(values, weight)
            for i in range(len(values)):
                votes[i] += (correct * prediction[i])
        
        for i in range(len(values)):
            if votes[i] > 0:
                predictions[i] = 1
            else:
                predictions[i] = -1

        return predictions
    
    def average(self, values, labels, epochs, r):
        ''' 
        Average perceptron algorithm

        Returns
        -------
        Average learned weight vector
        '''
        weight = [0] * self.N
        avg_weight = [0] * self.N
        for _ in range(epochs):
            for row in range(self.M):
                predict = 0
                epoch_weight = weight.copy()
                # Generate prediction and potential new epoch weight
                for column in range(len(values[row])):
                    predict += (values[row][column] * weight[column])
                    epoch_weight[column] = weight[column] + (r * labels[row] * values[row][column])
                    avg_weight[column] += weight[column]

                # Was prediction incorrect?
                if labels[row] * predict <= 0:
                    weight = epoch_weight

        self.prediction = self.standard_prediction(values, avg_weight)

        return avg_weight

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

def standard_perceptron_test(values, labels, test_values, test_labels):
    print("RUNNING STANDARD PERCEPTRON...")
    print("TRAIN DATA")
    perceptron = Perceptron(values, labels, Perceptron.standard, 10, 1)
    print(f'Learned weight vector is : {perceptron.weight}')

    total_examples = len(labels)
    correct_predictions = 0
    for i in range(total_examples):
        if perceptron.prediction[i] == labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

    print("TEST DATA")
    total_examples = len(test_labels)
    correct_predictions = 0
    test_predictions = perceptron.standard_prediction(test_values, perceptron.weight)
    for i in range(total_examples):
        if test_predictions[i] == test_labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

def voted_perceptron_test(values, labels, test_values, test_labels):
    print("RUNNING VOTED PERCEPTRON...")
    perceptron = Perceptron(values, labels, Perceptron.voted, 10, 1)
    for i in range(5):
        print(f'Learned k{i} weight vector is : {perceptron.weight[i]}')
    for i in range(5):
        print(f'Learned k{len(perceptron.weight) + i - 5} weight vector is : {perceptron.weight[len(perceptron.weight) + i - 5]}')

    total_examples = len(labels)
    correct_predictions = 0
    for i in range(total_examples):
        if perceptron.prediction[i] == labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

    print("TEST DATA")
    total_examples = len(test_labels)
    correct_predictions = 0
    test_predictions = perceptron.voted_prediction(test_values, perceptron.weight)
    for i in range(total_examples):
        if test_predictions[i] == test_labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

def average_perceptron_test(values, labels, test_values, test_labels):
    print("RUNNING AVERAGE PERCEPTRON...")
    perceptron = Perceptron(values, labels, Perceptron.average, 10, 1)
    print(f'Learned weight vector is : {perceptron.weight}')

    total_examples = len(labels)
    correct_predictions = 0
    for i in range(total_examples):
        if perceptron.prediction[i] == labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

    print("TEST DATA")
    total_examples = len(test_labels)
    correct_predictions = 0
    test_predictions = perceptron.standard_prediction(test_values, perceptron.weight)
    for i in range(total_examples):
        if test_predictions[i] == test_labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

def main():
    traindata = read_csv_line_prepend_bias("Perceptron/bank-note/train.csv", [0,1,2,3,4])
    testdata  = read_csv_line_prepend_bias("Perceptron/bank-note/test.csv", [0,1,2,3,4])

    standard_perceptron_test(traindata[0], traindata[1], testdata[0], testdata[1])
    voted_perceptron_test(traindata[0], traindata[1], testdata[0], testdata[1])
    average_perceptron_test(traindata[0], traindata[1],  testdata[0], testdata[1])

if __name__ == "__main__":
    main()