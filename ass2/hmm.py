#!/usr/bin/env python3

# hmm.py: Implementation of a hidden markov model for the Umbrella domain.
# Author: Harald Husum
# Date: 02.03.2016
from typing import List

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / vector.sum()


def forward(
    sensor_model: np.ndarray, transition_model: np.ndarray, message: np.ndarray
) -> np.ndarray:
    # As defined in Russel Norvig Equation 15.12
    return normalize(sensor_model @ transition_model.T @ message)


def backward(
    sensor_model: np.ndarray, transition_model: np.ndarray, message: np.ndarray
) -> np.ndarray:
    # As defined in Russel Norvig Equation 15.13
    return transition_model @ sensor_model @ message


def forward_backward(evidence: np.ndarray, prior: np.ndarray) -> List[np.ndarray]:
    """The forward–backward algorithm for smoothing: computing posterior prob-
    abilities of a sequence of states given a sequence of observations.

    As defined in Russel Norvig Figure 15.4.

    :param evidence: A vector of evidence values for steps 1, ..., t
    :param prior: The prior distribution on the initial state, P(X₀)
    :return: smoothed estimates
    """
    # Model matrices
    transition_model = np.array([[0.7, 0.3], [0.3, 0.7]])
    u_true = np.array([[0.9, 0.0], [0.0, 0.2]])
    u_false = np.array([[0.1, 0.0], [0.0, 0.8]])

    # Initializing local variables for algorithm
    forward_messages = [np.array([[0.0], [0.0]]) for _ in range((len(evidence) + 1))]
    smoothed_estimates = [np.array([[0.0], [0.0]]) for _ in range(len(evidence))]
    backward_message = np.array([[1.0], [1.0]])

    # Here we pretty much follow the book
    forward_messages[0] = prior
    for t in range(1, len(evidence) + 1):
        sensor_model = u_true if evidence[t - 1] else u_false
        forward_messages[t] = forward(
            sensor_model=sensor_model,
            transition_model=transition_model,
            message=forward_messages[t - 1],
        )

    # Uncomment following line for first backward_message, if desirable
    # print(backward_message)

    for i in range(len(evidence) - 1, -1, -1):
        smoothed_estimates[i] = normalize(forward_messages[i + 1] * backward_message)
        sensor_model = u_true if evidence[i] else u_false
        backward_message = backward(
            sensor_model=sensor_model,
            transition_model=transition_model,
            message=backward_message,
        )

        # Uncomment following line for the remaining backward_messages, if desirable
        # print(backward_message)

    return smoothed_estimates


def main() -> None:
    # Simple framework to generate test data
    original_message = np.array([[0.5], [0.5]])
    evidence = np.array([True, True, False, True, True])

    fb = forward_backward(evidence, original_message)

    print("Estimates:\n")
    for estimate in fb:
        print(estimate, end="\n\n")


if __name__ == "__main__":
    main()
