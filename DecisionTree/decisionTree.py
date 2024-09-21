from math import log2
from math import inf

# https://stackoverflow.com/a/28015122
class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def count_labels(values):
    total = 0
    for k in values:
        total += k
    
    return total

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

def entropy(k_labels):
    k_labels.pop("filter", None)
    values = k_labels.values()
    total = count_labels(values)

    sum = 0
    for k in values:
        if k == 0:
            sum += 0
        else:
            sum += (k / total) * log2(k / total)
    
    return (-sum, total)

def ME(k_labels):
    k_labels.pop("filter", None)
    values = k_labels.values()
    total = count_labels(values)

    return (min(values) / total, total)
    
def gini(k_labels):
    k_labels.pop("filter", None)
    values = k_labels.values()
    total = count_labels(values)

    sum = 1
    for k in values:
        sum -= (k / total) ** 2
    
    return (sum, total)

def info_gain(purity, attribute, labels):
    subsets = create_subsets(attribute, labels)

    gain = purity[0]
    total_labels = purity[1]
    for subset in subsets:
        subset_labels = subsets[subset]
        temp_sub = subsets[subset]['filter']

        res = entropy(subset_labels)
        gain -= (res[1] / total_labels) * res[0]

        subsets[subset]['filter'] = temp_sub
    
    return (gain, subsets)

def split(attributes, labels, func=entropy):
    label_counts = create_label(labels)

    best_attribute = ''
    max_gain = -inf
    subsets_attribute = {}
    for attribute in attributes:
        res = info_gain(func(label_counts), attributes[attribute], labels)
        gain = res[0]
        # print(f"The gain for attribute {attribute} is {gain}")
        subset = res[1]
        if gain > max_gain:
            best_attribute = attribute
            max_gain = gain
            subsets_attribute = subset
    
    # print(f"Split on {best_attribute} which has subsets of {subsets_attribute}")
    return (best_attribute, subsets_attribute)

def ID3(s, attributes):
    labels = create_label(s)
    count = len(s)

    max_label = -inf
    common_label = ''
    # If all examples have same labels return leaf node with label or most common label
    for l in labels:
        if labels[l] == count:
            return Tree(l)
        if labels[l] > max_label:
            common_label = l
            max_label = labels[l]
    if len(attributes) == 0:
        return Tree(common_label)
    
    #Else
    A = split(attributes, s)
    name = A[0]
    A_subset = A[1]

    root = Tree(name)
    
    for v in A_subset:
        nl = []
        na = {}
        filter = A_subset[v]["filter"]
        ca = attributes.copy()
        ca.pop(name)
        for attr in ca:
            if attr not in na:
                na[attr] = []
            for i in filter:
                na[attr].append(attributes[attr][i])
        for i in filter:
            nl.append(s[i])
        
        if len(nl) == 0:
            root.add_child(Tree(common_label))
        else:
            root.add_child(ID3(nl, na))
    
    return root
        
def main():
    l = [0,0,1,1,0,0,0]
    a = {"x1":[0,0,0,1,0,1,0], "x2":[0,1,0,0,1,1,1], "x3":[1,0,1,0,1,0,0],"x4":[0,0,1,1,0,0,1]}
    a = ID3(l, a)

if __name__ == '__main__':
    main()