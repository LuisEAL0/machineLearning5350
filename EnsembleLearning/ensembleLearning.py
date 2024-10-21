from decisionTree import *
from predictDT import *
from random import randrange
from math import inf

class EnsembleLearning():
    '''TODO'''
    def __init__(self, values, labels, attributeNames,labelNames, medians, decisionTree):
        '''TODO'''
        self.values = values
        self.labels = labels
        self.attributeNames = attributeNames
        self.labelNames = labelNames
        self.medians = medians
        self.DT = decisionTree
        self.familyOfTrees = None

        name = ''
        self.examples = {}
        for attribute in values.keys():
            if attribute not in self.examples:
                self.examples[attribute] = []
            name = attribute
        self.N = len(values[name])
        self.votes = [None] * self.N

    def bagged(self, m, numberOfTrees):
        '''TODO'''
        t = 0
        self.familyOfTrees = [None] * numberOfTrees
        while t < numberOfTrees:
            examplesSize = int(m * self.N)
            examples = {}
            for attribute in self.examples.keys():
                if attribute not in examples:
                    examples[attribute] = []
            
            labelExamples = []
            for _ in range(examplesSize):
                rand_i = randrange(0, self.N)
                for attribute in self.values:
                    examples[attribute].append((self.values[attribute][rand_i]))
                labelExamples.append(self.labels[rand_i])
            self.familyOfTrees[t] = self.DT(labelExamples, examples, self.attributeNames, 0, inf)
            t += 1
        
        return self.familyOfTrees
    
    def forest(self, m, numberOfTrees, size):
        '''TODO'''
        t = 0
        self.familyOfTrees = [None] * numberOfTrees
        while t < numberOfTrees:
            examplesSize = int(m * self.N)
            examples = {}
            for attribute in self.examples.keys():
                if attribute not in examples:
                    examples[attribute] = []
            
            labelExamples = []
            for _ in range(examplesSize):
                rand_i = randrange(0, self.N)
                for attribute in self.values:
                    examples[attribute].append((self.values[attribute][rand_i]))
                labelExamples.append(self.labels[rand_i])
            self.familyOfTrees[t] = self.DT(labelExamples, examples, self.attributeNames, 0, inf, randomForest=True, randomForestSize=size)
            t += 1
    
        return self.familyOfTrees

    def vote(self, example, voters):
        '''TODO'''
        votes = {}

        for label in self.labelNames:
            votes[label] = 0
        
        for index in range(voters):
            if self.votes[index] is None:
                vote = predict(example, self.familyOfTrees[index], self.labelNames, self.medians)
                votes[vote] += 1
                self.votes[index] = vote
            else:
                votes[self.votes[index]] += 1
        
        voteCount = 0
        finalVote = ''
        for vote in votes:
            if votes[vote] > voteCount:
                voteCount = votes[vote]
                finalVote = vote
        
        return finalVote

def main():
    data = bankData(read_csv_asdict("../EnsembleLearning/bankData/train.csv"))
    # baggers = EnsembleLearning(data["values"], data["labels"], data["attributeNames"], data["labelNames"], data["medians"],ID3)
    # baggers.forest(.20, 2, 2)

    test_bank = []
    train_bank = []
    total = len(data["labels"])
    train_data = read_csv_line("../EnsembleLearning/bankData/train.csv")
    test_data = read_csv_line("../EnsembleLearning/bankData/test.csv")

    baggers = EnsembleLearning(data["values"], data["labels"], data["attributeNames"], data["labelNames"], data["medians"],ID3)
    baggers.forest(.20, 500, 6)

    for i in range(500):
        if i % 10 == 0:
            print("Trees" + str(i))
        correct = 0
        for row in train_data:
            for item in row:
                if item in data["medians"].keys():
                    row[item] = int(row[item])
                # if row[item] == "unknown":
                #     row[item] = missing_values[item]
            
            if row["label"] == baggers.vote(row, i + 1):
                correct += 1
        train_bank.append(correct/total)

        correct = 0
        baggers.votes = [None] * 5000
        for row in test_data:
            for item in row:
                if item in data["medians"].keys():
                    row[item] = int(row[item])
                # if row[item] == "unknown":
                #     row[item] = missing_values[item]

            if row["label"] == baggers.vote(row, i + 1):
                correct += 1
        test_bank.append(correct/total)
    
    print(train_bank)
    print(test_bank)

if __name__ == "__main__":
    main()