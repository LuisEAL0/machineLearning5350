import csv
from decisionTree import *

def read_csv_asdict(filepath):
    data_dict = {}

    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        
        for header in csv_reader.fieldnames:
            data_dict[header] = []

        for row in csv_reader:
            for header in csv_reader.fieldnames:
                data_dict[header].append(row[header])

    return data_dict

def read_csv_line(filepath):
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]

    return data

def test_car(data_dict, function):
    labels = data_dict["label"]
    data_dict.pop("label")
    attributes = data_dict
    car_attributes = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
    }
    car_labels = ["unacc", "acc", "good", "vgood"]

    test_car = []
    train_car = []
    total = len(labels)
    for tree_size in range(1,7):
        tree = ID3(labels, attributes, car_attributes, 0, tree_size, function)

        correct = 0
        train_data = read_csv_line("../DecisionTree/carData/train.csv")
        for row in train_data:
            if row["label"] == predict(row, tree, car_labels):
                correct += 1
        train_car.append(correct/total)

        correct = 0
        test_data = read_csv_line("../DecisionTree/carData/test.csv")
        for row in test_data:
            if row["label"] == predict(row, tree, car_labels):
                correct += 1
        test_car.append(correct/total)
    
    return (train_car, test_car)
             
def predict(example, tree, labels):
    if tree.branch:
        return predict(example, tree.children[0], labels)
    if tree.name in labels:
        return tree.name
    
    at_needed = tree.name 
    example_v = example[at_needed]

    for c in tree.children:
        if c.name == example_v:
            return predict(example, c, labels)

def main():
    gains = [entropy, gini, ME]
    res0 = test_car(read_csv_asdict("../DecisionTree/carData/train.csv"), gains[0])
    res1 = test_car(read_csv_asdict("../DecisionTree/carData/train.csv"), gains[1])
    res2 = test_car(read_csv_asdict("../DecisionTree/carData/train.csv"), gains[2])
    
    train0, train1, train2 = res0[0],res1[0],res2[0]
    test0, test1, test2 = res0[1],res1[1],res2[1]

    print(f"TRAINING DATA")
    print(f"{"SIZE":{" "}{"^"}{4}} | {"ENTROPY":{" "}{"^"}{9}} | {"GINI":{" "}{"^"}{8}} | {"ME":{" "}{"^"}{8}}|")
    for i in range(len(train0)):
        print(f"{i+1:{" "}{">"}{4}} | {(train0[i]):{" "}{"^"}{9}.3f} | {(train1[i]):{" "}{"^"}{8}.3f} | {(train2[i]):{" "}{"^"}{7}.3f} |")

    print(f"\nTESTING DATA")
    print(f"{"SIZE":{" "}{"^"}{4}} | {"ENTROPY":{" "}{"^"}{9}} | {"GINI":{" "}{"^"}{8}} | {"ME":{" "}{"^"}{8}}|")
    for i in range(len(test0)):
        print(f"{i+1:{" "}{">"}{4}} | {(test0[i]):{" "}{"^"}{9}.3f} | {(test1[i]):{" "}{"^"}{8}.3f} | {(test2[i]):{" "}{"^"}{7}.3f} |")

if __name__ == '__main__':
    main()