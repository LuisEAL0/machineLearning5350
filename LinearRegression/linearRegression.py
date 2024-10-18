from math import inf

class LinearRegression:
    def __init__(self, values: list, labels: list):
        self.values = values # Needs 1 to account for bias
        self.labels = labels
        self.weight = [[0] * len(labels)] * len(values) # Needs a bias
        self.gradient = [0] * len(labels)
        self.loss = inf
    
    def prediction(self):
        prediction = [0] * len(self.labels)
        for i in range(len(self.values)):
            for j in range(len(prediction)):
                prediction[j] += self.values[i][j] * self.weight[i][j]
        return prediction

    def updateLoss(self):
        loss = 0
        prediction = self.prediction()

        for i in range(len(self.labels)):
            loss += (self.labels[i] - prediction[i]) ** 2

        loss = .5 * loss
        self.loss = loss

        return self.loss

    def updateGradient(self):
          grad = 0
          for i in range(len(self.values)):
            for j in range(len(self.labels)):
                self.gradient[j] = (self.labels[i] - (self.weight[i][j] * self.values[i][j])) * self.values[i][j] * -1
    
    def batchGradient(epochs):
        for t in epochs:
            ...



def main():
    lr = LinearRegression([[31.5, 36.2, 43.1, 27.6], [6, 2, 0, 2]], [21, 25, 18, 30])
    print(lr.updateLoss())

if __name__ == "__main__":
    main()