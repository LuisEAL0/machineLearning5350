from math import log2
from math import inf
import pprint

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

def create_subsets(attr_name, attribute, labels, attr_set):
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
    
    for sub in subsets:
        if sub not in attr_set[attr_name]:
            index -= len(subsets[sub]["filter"])
    
    for sub in subsets:
        if sub in attr_set[attr_name]:
            subsets[sub]["proportion"] = len(subsets[sub]["filter"]) / index

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
    k_labels.pop("proportion", None)
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
    k_labels.pop("proportion", None)
    values = k_labels.values()
    total = count_labels(values)

    return (min(values) / total, total)
    
def gini(k_labels):
    k_labels.pop("filter", None)
    k_labels.pop("proportion", None)
    values = k_labels.values()
    total = count_labels(values)

    sum = 1
    for k in values:
        sum -= (k / total) ** 2
    
    return (sum, total)

def info_gain(purity, attribute, labels, func, a_s, attr_name):
    subsets = create_subsets(attr_name, attribute, labels, a_s)

    gain = purity[0]
    total_labels = purity[1]

    missing_labels = {}
    bad_value = ''
    for sub_v in subsets:
        if sub_v not in a_s[attr_name]:
            bad_value = sub_v
            for label in subsets[sub_v]:
                missing_labels[label] = subsets[sub_v][label]
    if bad_value != '':
        subsets.pop(bad_value)
    
    for label in missing_labels:
        if label != "filter":
            for sub_v in subsets:
                if missing_labels[label] != 0:
                    subsets[sub_v][label] += missing_labels[label] * subsets[sub_v]["proportion"]

    for subset in subsets:
        subset_labels = subsets[subset]
        temp_sub = subsets[subset]['filter']

        res = func(subset_labels)
        gain -= (res[1] / total_labels) * res[0]

        subsets[subset]['filter'] = temp_sub
    
    return (gain, subsets)

def split(attributes, labels, a_s, func=entropy):
    label_counts = create_label(labels)

    best_attribute = ''
    max_gain = -inf
    subsets_attribute = {}
    e = func(label_counts)
    print(f"The total purity of the system is: {e[0]:.3f}\n")
    for attribute in attributes:
        res = info_gain(e, attributes[attribute], labels, func, a_s, attribute)
        gain = res[0]
        print(f"The gain for attribute {attribute} is {gain:.3f}")
        subset = res[1]
        if gain > max_gain:
            best_attribute = attribute
            max_gain = gain
            subsets_attribute = subset
    
    print(f"\nSplit on {best_attribute} which has subsets of:")
    pprint.pprint(subsets_attribute)
    print()
    return (best_attribute, subsets_attribute)

def ID3(s, attributes, attribute_set, call, func=entropy):
    print(f'Call {call}')
    print(f'Labels {pprint.pformat(s)}')
    pprint.pprint(attributes)
    print()
    labels = create_label(s)
    count = len(s)

    max_label = -inf
    common_label = ''
    # If all examples have same labels return leaf node with label or most common label
    for l in labels:
        if labels[l] == count:
            print(f"Pick label {l}\n")
            return Tree(l)
        if labels[l] > max_label:
            common_label = l
            max_label = labels[l]
    if len(attributes) == 0:
        print(f"Pick label {common_label}\n")
        return Tree(common_label)
    
    #Else
    A = split(attributes, s, attribute_set, func)
    name = A[0]
    A_subset = A[1]

    root = Tree(name)
    
    for pv in attribute_set[name]:
        if pv not in A_subset:
            A_subset[pv] = {"filter":[]}

    for v in A_subset:
        v_node = Tree(v)
        root.add_child(v_node)
        print("Current subtree looks like:")
        tree_traversal(root, 0, 0)
        print()
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
            v_node.add_child(Tree(common_label))
        else:
            v_node.add_child(ID3(nl, na, attribute_set,call + 1,func))

    return root

def tree_traversal(node, ident, level):
    print(f'{str(level) + ":" + str(node.__repr__()):>{ident}}')
    for c in node.children:
        tree_traversal(c, ident + 3, level + 1)

def main():
    l = [1,0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    a = {"Outlook":["Missing","S", "S", "O", "R", "R", "R", "O", "S", "S", "R", "S", "O", "O", "R"],
         "Temperature":["M","H", "H", "H", "M", "C", "C", "C", "M", "C", "M", "M", "M", "H", "M"],
         "Humidity":["N","H", "H", "H", "H", "N", "N", "N", "H", "N", "N", "N", "H", "N", "H"],
         "Wind":["W","W", "S", "W", "W", "W", "S", "S", "W", "W", "W", "S", "S", "W", "S"]}
    a_s = {"Outlook": ["S", "O", "R"], "Temperature": ["H", "M", "C"], "Humidity":["H", "N", "L"], "Wind":["W", "S"]}
    a= ID3(l, a, a_s, 0)
    print("Final tree looks like:")
    tree_traversal(a, 0, 0)
if __name__ == '__main__':
    main()