#!/usr/bin/env python3

# decision.py: Implementation of the decision tree learning algorithm and
# 	variations on the Importance subroutine.
# Author: Harald Husum
# Date: 20.04.2016

import random
import math
import sys


class Node:
    def __init__(self, ndtype, attribute=-1):
        self.ndtype = ndtype
        self.attribute = attribute
        self.children = []

    def classify_example(self, example):
        if self.ndtype == "test":
            return self.children[example[self.attribute] - 1].classify_example(example)
        elif self.ndtype == "1":
            return 1
        else:
            return 2

    def print_node(self, indentation):
        prstr = ""
        for i in range(indentation):
            prstr += "\t"
        prstr += "NODE(" + repr(self.ndtype) + ")"
        if self.ndtype == "test":
            prstr += ": " + repr(self.attribute)
        print(prstr)

    def print_tree(self, indentation):
        self.print_node(indentation)
        for c in self.children:
            c.print_tree(indentation + 1)


def plurality_value(examples):
    c1 = 0
    c2 = 0
    for e in examples:
        if e[7] == 1:
            c1 += 1
        else:
            c2 += 1

    if c1 >= c2:
        return Node("1")
    else:
        return Node("2")


def uniform_class(examples):
    uniform = True
    for e in examples[1:]:
        if e[7] != examples[0][7]:
            uniform = False
    return uniform


def importance_rand(attribute, examples):
    del attribute
    del examples
    return random.random()


def b(q):
    if q in [0, 1]:
        return 0
    return -(q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))


def set_entr(examples):
    p = 0
    for e in examples:
        if e[7] == 1:
            p += 1
    return b(p / len(examples))


def importance_entr(attribute, examples):
    goal = set_entr(examples)
    l1 = []
    l2 = []
    for e in examples:
        if e[attribute] == 1:
            l1.append(e)
        else:
            l2.append(e)
    remainder = (len(l1) / len(examples)) * set_entr(l1) + (
        len(l2) / len(examples)
    ) * set_entr(l2)
    gain = goal - remainder
    return gain


def importance(attribute, examples, imp_type):
    if imp_type:
        return importance_entr(attribute, examples)
    else:
        return importance_rand(attribute, examples)


def decision_tree_learning(examples, attributes, parent_examples, imp_type):
    if len(examples) == 0:
        return plurality_value(parent_examples)

    elif uniform_class(examples):
        if examples[0][7] == 1:
            tree = Node("1")
        else:
            tree = Node("2")
        return tree

    elif len(attributes) == 0:
        return plurality_value(examples)

    else:
        ms_attribute = -1
        val = -1
        for a in attributes:
            aval = importance(a, examples, imp_type)
            if aval > val:
                ms_attribute = a
                val = aval

        tree = Node("test", ms_attribute)

        for v in [1, 2]:
            vexs = []
            for e in examples:
                if e[ms_attribute] == v:
                    vexs.append(e)
            rem_attr = list(attributes)
            rem_attr.remove(ms_attribute)

            subtree = decision_tree_learning(vexs, rem_attr, examples, imp_type)

            tree.children.append(subtree)

        return tree


def extract_examples(file):
    with open(file, "r") as f:
        examples = []
        for line in f:
            example = []
            for c in line:
                try:
                    example.append(int(c))
                except ValueError:
                    pass
            examples.append(example)
    return examples


def write_tree(tree, file):
    std = sys.stdout
    sys.stdout = open(file, "w")
    tree.print_tree(0)
    sys.stdout.close()
    sys.stdout = std


def main():
    examples = extract_examples("training.txt")
    test_data = extract_examples("test.txt")
    attributes = [0, 1, 2, 3, 4, 5, 6]
    tree_rnd = decision_tree_learning(examples, attributes, [], False)
    tree_ent = decision_tree_learning(examples, attributes, [], True)

    write_tree(tree_rnd, "rndtree.txt")
    write_tree(tree_ent, "enttree.txt")

    errs = []
    for i in range(100):
        tree_rnd = decision_tree_learning(examples, attributes, [], False)
        rc = 0
        for e in test_data:
            if e[7] != tree_rnd.classify_example(e):
                rc += 1
        errs.append(rc)
    print("Low:")
    print(min(errs))
    print("Average:")
    print(sum(errs) / len(errs))
    print("High:")
    print(max(errs))

    rc = 0
    ec = 0
    print("Rnd:\tEnt:")
    for e in test_data:
        if e[7] != tree_rnd.classify_example(e):
            rc += 1
        if e[7] != tree_ent.classify_example(e):
            ec += 1
        print(
            str(e[7])
            + str(tree_rnd.classify_example(e))
            + "\t"
            + str(e[7])
            + str(tree_ent.classify_example(e))
        )
    print("\n")
    print("Total error:")
    print(f"{rc}\t{ec}")
    print("of\tof")
    print(f"{len(test_data)}\t{len(test_data)}")


if __name__ == "__main__":
    main()
