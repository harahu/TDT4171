#!/usr/bin/env python3

# decision.py: Implementation of the decision tree learning algorithm and
# 	variations on the importance subroutine.
# Author: Harald Husum
# Date: 20.04.2016
# Updates in 2020 just for fun

import random
import math
import statistics
import functools
from typing import Any
from typing import List
from typing import Tuple

VALUE_OPTIONS = {1, 2}


class Dataset:
    def __init__(self, examples: List[Tuple[List[int], int]]):
        self.examples: List[Tuple[List[int], int]] = examples

    @classmethod
    def from_file(cls, file_path: str) -> "Dataset":
        return cls(examples=load_dataset(file_path))

    @property
    def features(self) -> List[List[int]]:
        return [e[0] for e in self.examples]

    @property
    def labels(self) -> List[int]:
        return [e[1] for e in self.examples]


class Node:
    def __init__(self, ndtype: str, attribute: int = -1):
        self.ndtype: str = ndtype
        self.attribute: int = attribute
        self.children: List[Node] = []

    def __str__(self):
        return "\n".join(self.tree_strings())

    def classify_example(self, example: List[int]) -> int:
        if self.ndtype == "test":
            return self.children[example[self.attribute] - 1].classify_example(example)
        elif self.ndtype == "1":
            return 1
        else:
            return 2

    def node_string(self) -> str:
        prstr = f"NODE({self.ndtype})"
        if self.ndtype == "test":
            prstr += f": {self.attribute}"
        return prstr

    def tree_strings(self) -> List[str]:
        child_strings = [f"\t{s}" for c in self.children for s in c.tree_strings()]
        return [self.node_string(), *child_strings]


def plurality_value(examples: List[Tuple[List[int], int]]) -> Node:
    labels = [e[1] for e in examples]
    plurality_class = statistics.mode(labels)
    return Node(str(plurality_class))


def uniform_class(examples: List[Tuple[List[int], int]]) -> bool:
    labels = [e[1] for e in examples]
    return len(set(labels)) == 1


def b(q: float) -> float:
    if q in [0, 1]:
        return 0
    return -(q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))


def set_entropy(examples: List[Tuple[List[int], int]]) -> float:
    p = 0
    for e in examples:
        if e[1] == 1:
            p += 1
    return b(p / len(examples))


def importance_entropy(attribute: int, examples: List[Tuple[List[int], int]]) -> float:
    goal = set_entropy(examples)
    value_splits = [
        [e for e in examples if e[0][attribute] == v] for v in VALUE_OPTIONS
    ]
    remainder = sum(
        (len(split) / len(examples)) * set_entropy(split) for split in value_splits
    )
    gain = goal - remainder
    return gain


def importance_random(attribute: Any, examples: Any) -> float:
    del attribute
    del examples
    return random.random()


def decision_tree_learning(
    examples: List[Tuple[List[int], int]],
    considered_attributes: List[int],
    parent_examples: List[Tuple[List[int], int]],
    random_importance: bool = False,
) -> Node:
    importance_fn = importance_random if random_importance else importance_entropy

    if len(examples) == 0:
        return plurality_value(parent_examples)

    elif uniform_class(examples):
        return Node(str(examples[0][1]))

    elif len(considered_attributes) == 0:
        return plurality_value(examples)

    else:
        most_significant_attribute = max(
            considered_attributes,
            key=functools.partial(importance_fn, examples=examples),
        )
        tree = Node("test", most_significant_attribute)

        for value in VALUE_OPTIONS:
            value_examples = [
                e for e in examples if e[0][most_significant_attribute] == value
            ]

            subtree = decision_tree_learning(
                examples=value_examples,
                considered_attributes=[
                    a for a in considered_attributes if a != most_significant_attribute
                ],
                parent_examples=examples,
                random_importance=random_importance,
            )

            tree.children.append(subtree)

        return tree


def load_dataset(file_path: str) -> List[Tuple[List[int], int]]:
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            example = [int(n) for n in line.rstrip("\n").split("\t")]
            examples.append((example[:-1], example[-1]))
    return examples


def write_tree(tree: Node, file: str) -> None:
    with open(file, "w") as f:
        f.write(str(tree))


def main() -> None:
    training_set = Dataset.from_file("training.txt")
    test_set = Dataset.from_file("test.txt")
    attributes = [0, 1, 2, 3, 4, 5, 6]
    tree_rnd = decision_tree_learning(
        examples=training_set.examples,
        considered_attributes=attributes,
        parent_examples=[],
        random_importance=True,
    )
    tree_ent = decision_tree_learning(
        examples=training_set.examples,
        considered_attributes=attributes,
        parent_examples=[],
        random_importance=False,
    )

    write_tree(tree_rnd, "rndtree.txt")
    write_tree(tree_ent, "enttree.txt")

    errs = []
    for i in range(100):
        tree_rnd = decision_tree_learning(
            examples=training_set.examples,
            considered_attributes=attributes,
            parent_examples=[],
            random_importance=True,
        )
        rc = 0
        for e in test_set.examples:
            if e[1] != tree_rnd.classify_example(e[0]):
                rc += 1
        errs.append(rc)
    print(f"Low: {min(errs)}")
    print(f"Average: {sum(errs) / len(errs)}")
    print(f"High: {max(errs)}")

    rc = 0
    ec = 0
    print("Rnd:\tEnt:")
    for e in test_set.examples:
        if e[1] != tree_rnd.classify_example(e[0]):
            rc += 1
        if e[1] != tree_ent.classify_example(e[0]):
            ec += 1
        print(
            f"{e[1]}{tree_rnd.classify_example(e[0])}\t{e[1]}{tree_ent.classify_example(e[0])}"
        )
    print()
    print("Total error:")
    print(f"{rc}\t{ec}")
    print("of\tof")
    print(f"{len(test_set.examples)}\t{len(test_set.examples)}")


if __name__ == "__main__":
    main()
