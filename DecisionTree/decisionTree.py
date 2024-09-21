from math import log2
from math import inf

def bool_table1(x2, x4):
    return x4 & ~x2

def entropy(k_labels):
    values = k_labels.values()
    total = 0
    for k in values:
        total += k
    
    sum = 0
    for k in values:
        if k == 0:
            sum += 0
        else:
            sum += (k / total) * log2(k / total)
    
    return (-sum, total)

def info_gain(total_entropy, attribute, labels):
    subsets = create_subsets(attribute, labels)

    gain = total_entropy[0]
    total_labels = total_entropy[1]
    for subset in subsets:
        subset_labels = subsets[subset]
        res = entropy(subset_labels)
        gain -= (res[1] / total_labels) * res[0]
    
    return gain

def create_subsets(attribute, labels):
    subsets = {}
    index = 0

    for item in attribute:
        if item not in subsets:
            subsets[item] = {}
            for label in labels:
                subsets[item][label] = 0
                
        subsets[item][labels[index]] += 1
        index += 1

    return subsets

def create_label(labels):
    label_dict = {}
    for label in labels:
        if label not in label_dict:
            label_dict[label] = 0
        label_dict[label] += 1

    return label_dict

def split(attributes, labels):
    label_counts = create_label(labels)

    best_attribute = ''
    max_gain = -inf
    for attribute in attributes:
        gain = info_gain(entropy(label_counts), attributes[attribute], labels)
        if info_gain(entropy(label_counts), attributes[attribute], labels) > max_gain:
            best_attribute = attribute
            max_gain = gain
    
    return best_attribute


def ID3(s, attributes, label):
    ...

def main():
    data = {
        "x1": [0, 0, 0, 1, 0, 1, 0],
        "x2": [0, 1, 0, 0, 1, 1, 1],
        "x3": [1, 0, 1, 0, 1, 0, 0],
        "x4": [0, 0, 1, 1, 0, 0, 0]
    }
    labels = {
        "y": [0, 0, 1, 1, 0, 0, 0]
    }
    split(data, labels["y"])

if __name__ == '__main__':
    main()