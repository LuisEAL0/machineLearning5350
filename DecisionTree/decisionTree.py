from math import log2
from math import inf

# https://stackoverflow.com/a/28015122
class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None, data=None):
        self.name = name
        self.data = data
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name
    
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def bool_table1(x2, x4):
    return x4 & ~x2

def entropy(k_labels):
    k_labels.pop("filter", None)
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
        temp_sub = subsets[subset]['filter']

        res = entropy(subset_labels)
        gain -= (res[1] / total_labels) * res[0]
        
        subsets[subset]['filter'] = temp_sub
    
    return (gain, subsets)

def create_subsets(attribute, labels):
    subsets = {}
    index = 0

    for item in attribute:
        if item not in subsets:
            subsets[item] = {"filter":[]}
            for label in labels:
                subsets[item][label] = 0
        
        subsets[item]["filter"].append(index)
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
    subsets_attribute = {}
    for attribute in attributes:
        res = info_gain(entropy(label_counts), attributes[attribute], labels)
        gain = res[0]
        subset = res[1]
        if gain > max_gain:
            best_attribute = attribute
            max_gain = gain
            subsets_attribute = subset
    
    return (best_attribute, subsets_attribute)

def ID3(s, attributes, label):
    labels = create_label(s)
    count = len(s)

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
    ID3(labels["y"], data, [0, 1])

if __name__ == '__main__':
    main()