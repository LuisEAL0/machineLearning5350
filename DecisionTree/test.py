import csv
from decisionTree import *

def readcsv(filepath):
    data_dict = {}

    # Open the CSV file
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)  # Read the file as a dictionary
        
        # Initialize the dictionary keys based on the headers
        for header in csv_reader.fieldnames:
            data_dict[header] = []

        # Populate the dictionary with column values
        for row in csv_reader:
            for header in csv_reader.fieldnames:
                data_dict[header].append(row[header])

    return data_dict

def test_car(data_dict):
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

    test_data = readcsv("../DecisionTree/carData/test.csv")
    for tree_size in range(1,7):
        tree = ID3(labels, attributes, car_attributes, 0, tree_size)
        

def main():
    test_car(readcsv("../DecisionTree/carData/train.csv"))

if __name__ == '__main__':
    main()