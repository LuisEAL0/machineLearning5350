from math import inf
from math import sqrt
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
        """
        self.values = values
        self.labels = labels
        self.prediction = [0] * len(self.labels)
        self.weight = [0] * (len(values[0]))
        self.gradient = [0] * len(values[0])
        self.loss = inf
    
    def createPrediction(self):
        '''Creates a prediction vector using the self values and weight vectors'''
        for i in range(len(self.values)):
            self.prediction[i] = 0
            for j in range(len(self.weight)):
                self.prediction[i] += self.values[i][j] * self.weight[j]

        return self.prediction

    def updateLoss(self):
        '''Calculates the total loss of current prediction'''
        loss = 0
        for i in range(len(self.labels)):
            loss += (self.labels[i] - self.prediction[i]) ** 2
        self.loss = .5 * loss

        return self.loss

    def updateGradient(self):
        '''Calculates gradient of the current prediction'''
        for i in range(len(self.gradient)):
            self.gradient[i] = 0
            for j in range(len(self.values)):
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
        epochWeight = [0] * len(self.weight)
        weightDifference = [1] * len(self.weight)
        
        while(errorFunc(weightDifference) > (10 ** -6) or t > maxLoops):
            self.createPrediction()
            self.updateLoss()
            if self.loss > (10 ** 100):
                print("No convergence... Pick different r")
                break

            self.updateGradient()
            for i in range(len(self.weight)):
                epochWeight[i] = self.weight[i] - (r * self.gradient[i])
                weightDifference[i] = epochWeight[i] - self.weight[i]
            self.weight = epochWeight.copy()
            t += 1
        
        return self.weight
            
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
    testdata = read_csv_line_prepend_bias("./LinearRegression/concreteData/test.csv", ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Output"])

    lr = LinearRegression(traindata[0], traindata[1])
    test_lr = LinearRegression(testdata[0], testdata[1])
    test_lr.weight = lr.batchGradient(.001, euclidNorm).copy()

    # print(lr.createPrediction())
    # print(lr.updateLoss())

    print(test_lr.createPrediction())
    print(test_lr.labels)
    print(test_lr.updateLoss())

    # print(max(traindata[1]))
    # print(min(traindata[1]))

if __name__ == "__main__":
    main()