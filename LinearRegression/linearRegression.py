from math import inf
from math import sqrt

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
    
    def batchGradient(self, r):
        '''
        Batch Gradient Algorithm

        If loss grows to large will return latest prediction 
        
        Parameters
        ----------
        r : float
            learning rate
        '''

        t = 0
        epochWeight = [0] * len(self.weight)
        weightDifference = [1] * len(self.weight)
        
        while(euclidNorm(weightDifference) > (10 ** -6)):
            self.createPrediction()
            self.updateLoss()
            if self.loss > (10 ** 20):
                break

            self.updateGradient()
            for i in range(len(self.weight)):
                epochWeight[i] = self.weight[i] - (r * self.gradient[i])
                weightDifference[i] = epochWeight[i] - self.weight[i]
            self.weight = epochWeight.copy()
            t += 1
        
        return self.prediction
            
def euclidNorm(vector):
    '''Calculates the Euclidean Norm of a vector'''
    norm = 0
    for x in vector:
        norm += x ** 2
    
    return sqrt(norm)

def main():
    lr = LinearRegression([[1, 31.5, 6],
                           [1, 36.2, 2],
                           [1, 43.1, 0],
                           [1, 27.6, 2]], [21, 25, 18, 30])
    print(lr.batchGradient(.0001))
    print(lr.weight)

if __name__ == "__main__":
    main()