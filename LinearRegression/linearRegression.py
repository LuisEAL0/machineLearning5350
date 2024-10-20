from math import inf
from math import sqrt
from random import randrange
import numpy as np
import csv

class LinearRegression:
    ''' Linear Regression model to make predictions based on given data '''

    def __init__(self, values: list, labels: list):
        """
        Linear Regression takes in a set of attributes and labels then creates a weight vector
        that is used for prediction.

        Parameters
        ----------
        values : list
            attributes of a data set (should be preprocessed to include bias term)

        labels : list
            outputs of a data set
        
        Data
        ----
        prediction : list
            prediction using learned weight vector
        
        weight : list
            learned weight vector
        
        gradient : list
            gradient of prediction
        
        loss : integer
            total loss of the prediction
        
        lossEpochs : list
            total loss of the prediction over epochs
        
        M : integer
            Number of rows in data set
        
        N : integer
            Number of columns in data set
        """
        self.M          = len(values)
        self.N          = len(values[0])
        self.values     = values
        self.labels     = labels
        self.prediction = [0] * self.M
        self.weight     = [0] * self.N
        self.gradient   = [0] * self.N
        self.loss       = 0
        self.lossEpochs = []
    
    def createPrediction(self):
        '''Creates a prediction vector using the self values and weight vectors'''
        for i in range(self.M):
            self.prediction[i] = 0
            for j in range(self.N):
                self.prediction[i] += self.values[i][j] * self.weight[j]

        return self.prediction

    def updateLoss(self):
        '''Calculates the total loss of current prediction'''
        loss = 0
        for i in range(self.M):
            loss += (self.labels[i] - self.prediction[i]) ** 2
        self.loss = .5 * loss

        return self.loss

    def updateGradient(self):
        '''Calculates gradient of the current prediction'''
        for i in range(self.N):
            self.gradient[i] = 0
            for j in range(self.M):
                self.gradient[i] -= (self.labels[j] - self.prediction[j]) * self.values[j][i]

        return self.gradient
    
    def batchGradient(self, r, errorFunc):
        '''
        Batch Gradient Algorithm

        If loss grows to large will return latest weight 
        
        Parameters
        ----------
        r : float
            learning rate
        errorFunc : function
            function to calculate error between current and next weight
        
        Returns
        -------
        weight vector used to make predictions
        '''
        t = 0
        maxLoops = 1000000
        epochWeight = [0] * self.N
        weightDifference = [1] * self.N
        
        while(errorFunc(weightDifference) > (10 ** -6) or t > maxLoops):
            if self.loss > (10 ** 100):
                print("No convergence... Pick different r")
                break

            self.createPrediction()
            self.updateGradient()
            for i in range(self.N):
                epochWeight[i]      = self.weight[i] - (r * self.gradient[i])
                weightDifference[i] = epochWeight[i] - self.weight[i]
            self.weight = epochWeight.copy()
            t += 1

            self.lossEpochs.append(self.updateLoss())
        
        return self.weight

    def stochasticGradient(self, r, errorFunc):
        '''
        Stochastic Gradient Algorithm

        If loss grows to large will return latest weight 
        
        Parameters
        ----------
        r : float
            learning rate
        errorFunc : function
            function to calculate error between current and next weight
        
        Returns
        -------
        weight vector used to make predictions
        '''
        t = 0
        maxLoops = 1000000
        epochWeight = [0] * self.N
        weightDifference = [1] * self.N
        
        while(errorFunc(weightDifference) > (10 ** -6) or t > maxLoops):
            if self.loss > (10 ** 100):
                print("No convergence... Pick different r")
                break

            rand_i = randrange(0, self.M)
            rand_x = self.values[rand_i]

            self.prediction[rand_i] = 0
            for j in range(self.N):
                self.prediction[rand_i] += rand_x[j] * self.weight[j]

            self.gradient = [0] * len(rand_x)
            for j in range(self.N):
                self.gradient[j] -= (self.labels[rand_i] - self.prediction[rand_i]) * rand_x[j]
            
            for i in range(self.N):
                epochWeight[i]      = self.weight[i] - (r * self.gradient[i])
                weightDifference[i] = epochWeight[i] - self.weight[i]
            self.weight = epochWeight.copy()
            t += 1

            self.lossEpochs.append(self.updateLoss())
        
        return self.weight
    
    def analyticalW(self):
        '''Analytically solves for weight vector with (X. X^T)^-1 . X^T . Y'''
        xT = np.transpose(self.values)
        return (np.linalg.inv(xT @ self.values) @ xT @ self.labels).tolist()

def euclidNorm(vector):
    '''Calculates the Euclidean Norm of a vector'''
    norm = 0
    for x in vector:
        norm += x ** 2
    
    return sqrt(norm)

def read_csv_line_prepend_bias(filepath, attributes : list):
    '''
    Creates a values matrix (prepended with 1) and labels vector based off CSV file.
    Assumes final column are labels. 

    Parameters
    ----------
    filepath : string
        path to file
    attributes : list
        column names with last column assumed to be named 'Output'
    
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
                if attribute != "Output":
                    tempV.append(float(row[attribute]))
                else:
                    labels.append(float(row[attribute]))
            values.append(tempV)

    return [values, labels]

def main():
    traindata = read_csv_line_prepend_bias("./LinearRegression/concreteData/train.csv", ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Output"])
    testdata  = read_csv_line_prepend_bias("./LinearRegression/concreteData/test.csv", ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Output"])

if __name__ == "__main__":
    main()