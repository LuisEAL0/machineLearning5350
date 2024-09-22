import csv
import math
from decisionTree import *

def median(unsorted_list):
    unsorted_list.sort()
    return int(unsorted_list[len(unsorted_list)//2])

def numerics(attributes, bank_attributes):
    median_dict = {}
    for attr in bank_attributes:
        if bank_attributes[attr] == "numeric":
            median_dict[attr] = median(attributes[attr].copy())
    return median_dict

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

def most_common_label_replace(attributes, attribute_set):
    count_dict = {}
    missing_values = {}
    for attr in attributes:
        if attribute_set[attr] == "numeric" or ("unknown" not in attribute_set[attr]):
            continue
        if attr not in count_dict:
            count_dict[attr] = {}
            for v in attribute_set[attr]:
                count_dict[attr][v] = 0
        for i in attribute_set[attr]:
            for v in attributes[attr]:
                if v == i:
                    count_dict[attr][i] += 1
        attribute_set[attr].remove("unknown")
    
    for attr in attributes:
        if attr not in count_dict:
            continue
        replace = largest(count_dict, attr)
        missing_values[attr] = replace

    return missing_values

def largest(count_dict, attr):        
    largest_val = ''
    max_val = -inf

    for value in count_dict[attr]:
        if value != "unknown" and count_dict[attr][value] > max_val:
            largest_val = value
            max_val = count_dict[attr][value]
    return largest_val

def test_bank(data_dict, function):
    labels = data_dict["label"]
    data_dict.pop("label")
    attributes = data_dict.copy()
    bank_attributes = {
    "age": "numeric",
    "job": ["unknown","admin.", "unemployed", "management", "housemaid", "entrepreneur", "student",
            "blue-collar", "self-employed", "retired", "technician", "services"],
    "marital": ["married", "divorced", "single"],
    "education": ["unknown", "secondary", "primary", "tertiary"],
    "default": ["yes", "no"],
    "balance": "numeric",
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["unknown","telephone", "cellular"],
    "day": "numeric",
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration": "numeric",
    "campaign": "numeric",
    "pdays": "numeric",
    "previous": "numeric",
    "poutcome": ["unknown", "other", "failure", "success"]
    }
    bank_labels = ["yes", "no"]

    missing_values = most_common_label_replace(attributes, bank_attributes)
    median_dict = numerics(attributes, bank_attributes)
    for attr in bank_attributes:
        if bank_attributes[attr] == "numeric":
            bank_attributes[attr] = ["less", "greater"]
            for i in range(len(attributes[attr])):
                attributes[attr][i] = int(attributes[attr][i])
    
    test_bank = []
    train_bank = []
    total = len(labels)

    for tree_size in range(1,17):
        tree = ID3(labels, attributes, bank_attributes, 0, tree_size, function)

        correct = 0
        train_data = read_csv_line("../DecisionTree/bankData/train.csv")
        for row in train_data:
            for item in row:
                if item in median_dict.keys():
                    row[item] = int(row[item])
                if row[item] == "unknown":
                    row[item] = missing_values[item]
            
            if row["label"] == predict(row, tree, bank_labels, median_dict):
                correct += 1
        train_bank.append(correct/total)

        correct = 0
        test_data = read_csv_line("../DecisionTree/bankData/test.csv")
        for row in test_data:
            for item in row:
                if item in median_dict.keys():
                    row[item] = int(row[item])
                if row[item] == "unknown":
                    row[item] = missing_values[item]

            if row["label"] == predict(row, tree, bank_labels, median_dict):
                correct += 1
        test_bank.append(correct/total)
    
    return (train_bank, test_bank)
             
def predict(example, tree, labels, median={}):
    if tree.branch:
        return predict(example, tree.children[0], labels, median)
    if tree.name in labels:
        return tree.name
    
    at_needed = tree.name 
    example_v = example[at_needed]

    if type(example_v) is type(int()):
        if example_v > median[at_needed]:
            return predict(example, tree.children[1], labels, median)
        else:
            return predict(example, tree.children[0], labels, median)

    for c in tree.children:
        if c.name == example_v:
            return predict(example, c, labels, median)

def run_tests(gains, filepath, func):
    res0 = func(read_csv_asdict(filepath), gains[0])
    res1 = func(read_csv_asdict(filepath), gains[1])
    res2 = func(read_csv_asdict(filepath), gains[2])
    
    train0, train1, train2 = res0[0],res1[0],res2[0]
    test0, test1, test2 = res0[1],res1[1],res2[1]

    print(f"TRAINING DATA")
    print(f"{"SIZE":{" "}{"^"}{4}} | {"ENTROPY":{" "}{"^"}{9}} | {"GINI":{" "}{"^"}{8}} | {"ME":{" "}{"^"}{8}}|")
    for i in range(len(train0)):
        print(f"{i+1:{" "}{">"}{4}} | {(train0[i]):{" "}{"^"}{9}.3f} | {(train1[i]):{" "}{"^"}{8}.3f} | {(train2[i]):{" "}{"^"}{7}.3f} |")

    print(f"{"AVG":{" "}{">"}{4}} | {(avg(train0)):{" "}{"^"}{9}.3f} | {(avg(train1)):{" "}{"^"}{8}.3f} | {(avg(train2)):{" "}{"^"}{7}.3f} |")

    print(f"\nTESTING DATA")
    print(f"{"SIZE":{" "}{"^"}{4}} | {"ENTROPY":{" "}{"^"}{9}} | {"GINI":{" "}{"^"}{8}} | {"ME":{" "}{"^"}{8}}|")
    for i in range(len(test0)):
        print(f"{i+1:{" "}{">"}{4}} | {(test0[i]):{" "}{"^"}{9}.3f} | {(test1[i]):{" "}{"^"}{8}.3f} | {(test2[i]):{" "}{"^"}{7}.3f} |")
    
    print(f"{"AVG":{" "}{">"}{4}} | {(avg(test0)):{" "}{"^"}{9}.3f} | {(avg(test1)):{" "}{"^"}{8}.3f} | {(avg(test2)):{" "}{"^"}{7}.3f} |")

def avg(list):
    sum = 0
    size = 0
    for i in list:
        sum += i
        size += 1
    
    return sum / size

def main():
    gains = [entropy, gini, ME]
    run_tests(gains, "../DecisionTree/carData/train.csv", test_car)
    run_tests(gains, "../DecisionTree/bankData/train.csv", test_bank)

if __name__ == '__main__':
    main()