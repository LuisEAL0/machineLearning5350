import csv
import random

class SVM:
    ''' SVM model to make predictions based on given data '''

    def __init__(self, values: list, labels: list, svm_algorithm, epochs, C, r, step_size_func):
        """
        SVM takes in a set of values and labels then creates a weight vector
        that is used for prediction.

        Parameters
        ----------
        values : list
            attributes of a data set (should be preprocessed to include bias term)

        labels : list
            outputs of a data set (must be binary values of -1 and 1)
        
        svm_algorithm: function
            variant of SVM
        
        epochs: int
            number of epochs
        
        C: float
            regularization parameter
        
        r: float
            learning rate
        
        step_size_func: function
            function to determine step
        
        Internals
        ---------
        prediction : list
            prediction using learned weight vector
        
        weight : list
            learned weight vector(s) with counts
        
        M : integer
            Number of rows in data set
        
        N : integer
            Number of columns in data set
        
        r_naught : float
            initial learning rate
        """
        self.M = len(values)
        self.N = len(values[0])
        self.prediction = None
        self.r_naught = r
        self.weight = svm_algorithm(self, values, labels, epochs, C, r, step_size_func)

    def stochastic_sub_gradient_descent(self, values, labels, epochs, C, r, step_size_func):
        '''
        Stochastic Sub Gradient Descent algorithm for SVM.

        Returns
        -------
        Learned weight vector
        '''
        weight = [0] * self.N
        weight_naught = [0] * (self.N - 1)
        for t in range(epochs):
            shuffled_values, shuffled_labels = self.shuffle(values, labels)
            for row in range(self.M):
                predict = 0
                epoch_weight = weight.copy()
                epoch_naught = weight_naught.copy()

                # Generate prediction and potential new epoch weight and weight_naught
                for column in range(len(values[row])):
                    predict += (shuffled_values[row][column] * weight[column])
                    t0 = weight[column]
                    t1 = (step_size_func(r, t) * weight_naught[column - 1]) if column != 0 else 0
                    t2 = (C * self.N * step_size_func(self.r_naught, t) * shuffled_labels[row] * shuffled_values[row][column])
                    epoch_weight[column] = t0 - t1 + t2
                    if column != 0:
                        epoch_naught[column - 1] = (1 - step_size_func(self.r_naught, t)) * weight_naught[column - 1]

                # Did prediction violate the margin?
                if shuffled_labels[row] * predict <= 1:
                    weight = epoch_weight
                else:
                    weight_naught = epoch_naught
                    
        self.prediction = self.predictSubGradientSVM(values, weight)

        return weight

    def predictSubGradientSVM(self, values, weight):
        '''
        Creates prediction for the sub gradient SVM.

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

def step_size_func1(r, t):
    '''
    This function computes the step size based on the given parameters using the formula:
    step_size = r / (1 + t)

    Parameters
    ----------
        r : float
            The initial step size or learning rate.
        t : int
            The current iteration or time step.

    Returns
    -------
        The computed step size.
    '''
    return r / (1 + t)

def step_size_func2(r, t, a=1):
    '''
    This function computes the step size based on the given parameters using the formula:
    step_size = r / (1 + ((t * r) / a))

    Parameters
    ----------
        r : float
            The initial step size or learning rate.
        t : int
            The current iteration or time step.
        a : float
            The scaling factor for the step size.

    Returns
    -------
        The computed step size.
    '''
    return r / (1 + ((t * r) / a))

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


def SVMTest(values, labels, test_values, test_labels, C, r, epochs):
    print("RUNNING Stochastic Sub Gradient Descent...")
    print("TRAIN DATA")
    perceptron = SVM(values, labels, SVM.stochastic_sub_gradient_descent, epochs, C, r, step_size_func1)
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
    test_predictions = perceptron.predictSubGradientSVM(test_values, perceptron.weight)
    for i in range(total_examples):
        if test_predictions[i] == test_labels[i]:
            correct_predictions += 1
    print(f'The error rate is: {((1 - (correct_predictions/total_examples)) * 100):.2f}%\n')

def main():
    traindata = read_csv_line_prepend_bias("./bank-note/train.csv", [0,1,2,3,4])
    testdata  = read_csv_line_prepend_bias("./bank-note/test.csv", [0,1,2,3,4])

    SVMTest(traindata[0], traindata[1], testdata[0], testdata[1], 1, 0.01, 100)

if __name__ == '__main__':
    main()