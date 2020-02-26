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

    def classifyExample(self, example):
        if self.ndtype == "test":
            return self.children[example[self.attribute] - 1].classifyExample(example)
        elif self.ndtype == "1":
            return 1
        else:
            return 2

    def printNode(self, indentation):
        prstr = ""
        for i in range(indentation):
            prstr += "\t"
        prstr += "NODE(" + repr(self.ndtype) + ")"
        if self.ndtype == "test":
            prstr += ": " + repr(self.attribute)
        print(prstr)

    def printTree(self, indentation):
        self.printNode(indentation)
        for c in self.children:
            c.printTree(indentation + 1)


def pluralityValue(examples):
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


def uniformClass(examples):
    uniform = True
    for e in examples[1:]:
        if e[7] != examples[0][7]:
            uniform = False
    return uniform


def importanceRand(attribute, examples):
    return random.random()


def b(q):
    if q in [0, 1]:
        return 0
    return -(q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))


def setEntr(examples):
    p = 0
    for e in examples:
        if e[7] == 1:
            p += 1
    return b(p / len(examples))


def importanceEntr(attribute, examples):
    goal = setEntr(examples)
    l1 = []
    l2 = []
    for e in examples:
        if e[attribute] == 1:
            l1.append(e)
        else:
            l2.append(e)
    remainder = (len(l1) / len(examples)) * setEntr(l1) + (
        len(l2) / len(examples)
    ) * setEntr(l2)
    gain = goal - remainder
    return gain


def importance(attribute, examples, impType):
    if impType:
        return importanceEntr(attribute, examples)
    else:
        return importanceRand(attribute, examples)


def decisionTreeLearning(examples, attributes, parent_examples, impType):
    if len(examples) == 0:
        return pluralityValue(parent_examples)

    elif uniformClass(examples):
        if examples[0][7] == 1:
            tree = Node("1")
        else:
            tree = Node("2")
        return tree

    elif len(attributes) == 0:
        return pluralityValue(examples)

    else:
        msAttribute = -1
        val = -1
        for a in attributes:
            aval = importance(a, examples, impType)
            if aval > val:
                msAttribute = a
                val = aval

        tree = Node("test", msAttribute)

        for v in [1, 2]:
            vexs = []
            for e in examples:
                if e[msAttribute] == v:
                    vexs.append(e)
            remAttr = list(attributes)
            remAttr.remove(msAttribute)

            subtree = decisionTreeLearning(vexs, remAttr, examples, impType)

            tree.children.append(subtree)

        return tree


def extractExamples(file):
    f = open(file, "r")
    examples = []
    for l in f:
        example = []
        for c in l:
            try:
                example.append(int(c))
            except:
                pass
        examples.append(example)
    return examples
    f.close()


def writeTree(tree, file):
    std = sys.stdout
    sys.stdout = open(file, "w")
    tree.printTree(0)
    sys.stdout.close()
    sys.stdout = std


def main():
    examples = extractExamples("training.txt")
    testdata = extractExamples("test.txt")
    attributes = [0, 1, 2, 3, 4, 5, 6]
    treeRnd = decisionTreeLearning(examples, attributes, [], False)
    treeEnt = decisionTreeLearning(examples, attributes, [], True)

    writeTree(treeRnd, "rndtree.txt")
    writeTree(treeEnt, "enttree.txt")

    errs = []
    for i in range(100):
        treeRnd = decisionTreeLearning(examples, attributes, [], False)
        rc = 0
        for e in testdata:
            if e[7] != treeRnd.classifyExample(e):
                rc += 1
        errs.append(rc)
    print("Low:")
    print(str(min(errs)))
    print("Average:")
    print(str(sum(errs) / len(errs)))
    print("High:")
    print(str(max(errs)))

    rc = 0
    ec = 0
    print("Rnd:\tEnt:")
    for e in testdata:
        if e[7] != treeRnd.classifyExample(e):
            rc += 1
        if e[7] != treeEnt.classifyExample(e):
            ec += 1
        print(
            str(e[7])
            + str(treeRnd.classifyExample(e))
            + "\t"
            + str(e[7])
            + str(treeEnt.classifyExample(e))
        )
    print("\n")
    print("Total Error:")
    print(str(rc) + "\t" + str(ec))
    print("of\tof")
    print(str(len(testdata)) + "\t" + str(len(testdata)))


main()
