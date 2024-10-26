from utills import *
from random import uniform
import numpy as np
import matplotlib.pyplot as plt

generate_data_sets(["0", "1"])

# Perceptron
with open("output/training_set.txt") as f, open("output/training_set_labels.txt") as f2, open("output/test_set.txt") as f3, open("output/test_set_labels.txt") as f4, open("output/challenge_set.txt") as f5, open("output/challenge_set_labels.txt") as f6:
    training_set = f.readlines()
    training_set_labels = f2.readlines()

    test_set = f3.readlines()
    test_set_labels = f4.readlines()

    challenge_set = f5.readlines()
    challenge_set_labels = f6.readlines()

    # initialize weights
    weights = []
    for i in range(len(training_set[0].split("\t")) + 1):
        weights.append(uniform(0, 0.5))

    weights_untrained = weights.copy()
    # write out the initial weights to a text file
    with open("output/initial_weights.txt", "w") as f:
        for weight in weights_untrained:
            f.write(str(weight) + "\n")

    print("Untrained perceptron:\n")

    # get the output of the untrained perceptron on the training set
    output_training_set = []
    for input in training_set:
        input_arr = np.array([1] + input.split("\t"), dtype=float)
        net_input = calculate_net_input(weights, input_arr)
        output_training_set.append(1 if net_input >= 0 else 0)

    # calculate the error fraction
    false_positives = 0
    false_negatives = 0
    for i in range(len(output_training_set)):
        if output_training_set[i] == 1 and int(training_set_labels[i]) == 0:
            false_positives += 1
        elif output_training_set[i] == 0 and int(training_set_labels[i]) == 1:
            false_negatives += 1
    error_fraction_untrained_training = (
        false_positives + false_negatives) / len(training_set)
    print("On the training set:")
    print(f"Error fraction: {str(error_fraction_untrained_training)}\n")

    # get the output of the untrained perceptron on the test set
    output_untrained_test_set = []
    for input in test_set:
        input_arr = np.array([1] + input.split("\t"), dtype=float)
        net_input = calculate_net_input(weights, input_arr)
        output_untrained_test_set.append(1 if net_input >= 0 else 0)

    # calculate the error fraction, precision, recall, and F1 score
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    for i in range(len(output_untrained_test_set)):
        if output_untrained_test_set[i] == 1 and int(test_set_labels[i]) == 0:
            false_positives += 1
        elif output_untrained_test_set[i] == 0 and int(test_set_labels[i]) == 1:
            false_negatives += 1
        elif output_untrained_test_set[i] == 1 and int(test_set_labels[i]) == 1:
            true_positives += 1
    error_fraction_untrained_test = (
        false_positives + false_negatives) / len(test_set)
    precision_untrained_test = true_positives / \
        (true_positives + false_positives)
    recall_untrained_test = true_positives / (true_positives + false_negatives)
    f1_score_untrained_test = 2 * ((precision_untrained_test * recall_untrained_test) /
                                   (precision_untrained_test + recall_untrained_test))
    print("On the test set:")
    print(f"Error fraction: {str(error_fraction_untrained_test)}")
    print(f"Precision: {str(precision_untrained_test)}")
    print(f"Recall: {str(recall_untrained_test)}")
    print(f"F1 score: {str(f1_score_untrained_test)}\n")

    # train the perceptron
    weights_trained = np.array(weights)
    learning_rate = 0.001
    epochs = 5
    error_fractions = []
    for i in range(epochs):
        output_training_set = []
        for j in range(len(training_set)):
            input_arr = np.array(
                [1] + training_set[j].split("\t"), dtype=float)
            net_input = calculate_net_input(weights_trained, input_arr)
            output = 1 if net_input >= 0 else 0
            output_training_set.append(output)
            error = int(training_set_labels[j]) - output
            weights_trained += learning_rate * error * input_arr
        # calculate the error fraction
        false_positives = 0
        false_negatives = 0
        for j in range(len(output_training_set)):
            if output_training_set[j] == 1 and int(training_set_labels[j]) == 0:
                false_positives += 1
            elif output_training_set[j] == 0 and int(training_set_labels[j]) == 1:
                false_negatives += 1
        error_fraction = (false_positives + false_negatives) / \
            len(training_set)
        error_fractions.append(error_fraction)
        print(f"Epoch {i + 1}:")
        print(f"Error fraction: {str(error_fraction)}")
    # plot the error fraction over the epochs
    plt.plot(range(1, epochs + 1), error_fractions)
    plt.xlabel("Epoch")
    plt.ylabel("Error fraction")
    plt.title("Error fraction over epochs")
    plt.show()

    # get the output of the trained perceptron on the test set
    output_trained_test_set = []
    weights = weights_trained.copy()
    for input in test_set:
        input_arr = np.array([1] + input.split("\t"), dtype=float)
        net_input = calculate_net_input(weights, input_arr)
        output_trained_test_set.append(1 if net_input >= 0 else 0)
    # calculate the error fraction, precision, recall, and F1 score
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    for i in range(len(output_trained_test_set)):
        if output_trained_test_set[i] == 1 and int(test_set_labels[i]) == 0:
            false_positives += 1
        elif output_trained_test_set[i] == 0 and int(test_set_labels[i]) == 1:
            false_negatives += 1
        elif output_trained_test_set[i] == 1 and int(test_set_labels[i]) == 1:
            true_positives += 1
    error_fraction_trained_test = (
        false_positives + false_negatives) / len(test_set)
    precision_trained_test = true_positives / \
        (true_positives + false_positives)
    recall_trained_test = true_positives / (true_positives + false_negatives)
    f1_score_trained_test = 2 * ((precision_trained_test * recall_trained_test) /
                                 (precision_trained_test + recall_trained_test))
    # bar plot of the error fraction, precision, recall, and F1 score of the perceptron on the test set before and after training
    labels = ["Error fraction", "Precision", "Recall", "F1 score"]
    untrained = [error_fraction_untrained_test,
                 precision_untrained_test, recall_untrained_test, f1_score_untrained_test]
    trained = [error_fraction_trained_test,
               precision_trained_test, recall_trained_test, f1_score_trained_test]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, untrained, width, label="Untrained")
    rects2 = ax.bar(x + width/2, trained, width, label="Trained")
    ax.set_ylabel("Scores")
    ax.set_title(
        "Scores of the perceptron on the test set before and after training")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # create a range of bias weights
    weights = weights_trained.copy()
    trained_bias_weight = weights[0]
    new_bias_weights = np.linspace(
        trained_bias_weight - 10, trained_bias_weight + 10, 20)
    new_bias_weights = np.concatenate(
        (new_bias_weights[:11], [trained_bias_weight], new_bias_weights[11:]))

    # get the output of the trained perceptron on the test set for each new bias weight
    error_fractions = []
    precisions = []
    recalls = []
    f1_scores = []
    specificities = []
    for bias_weight in new_bias_weights:
        weights[0] = bias_weight
        output_trained_test_set = []
        for input in test_set:
            input_arr = np.array([1] + input.split("\t"), dtype=float)
            net_input = calculate_net_input(weights, input_arr)
            output_trained_test_set.append(1 if net_input >= 0 else 0)
        # calculate the error fraction
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        for i in range(len(output_trained_test_set)):
            if output_trained_test_set[i] == 1 and int(test_set_labels[i]) == 0:
                false_positives += 1
            elif output_trained_test_set[i] == 0 and int(test_set_labels[i]) == 1:
                false_negatives += 1
            elif output_trained_test_set[i] == 1 and int(test_set_labels[i]) == 1:
                true_positives += 1
        # calculate the error fraction, precision, recall, F1 score, and specificity
        error_fraction = (
            false_positives + false_negatives) / len(test_set)
        precision = 1 if (true_positives + false_positives) == 0 else true_positives / \
            (true_positives + false_positives)
        recall = 1 if (true_positives + false_negatives) == 0 else true_positives / \
            (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        specificity = 1 if (true_positives + false_positives) == 0 else true_positives / \
            (true_positives + false_positives)
        error_fractions.append(error_fraction)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        specificities.append(specificity)

    # plot the error fraction, precision, recall, and F1 score over the range of bias weights for the trained perceptron on the test set on the same plot
    plt.plot(new_bias_weights, error_fractions, label="Error fraction")
    plt.plot(new_bias_weights, precisions, label="Precision")
    plt.plot(new_bias_weights, recalls, label="Recall")
    plt.plot(new_bias_weights, f1_scores, label="F1 score")
    plt.xlabel("Bias weight")
    plt.ylabel("Scores")
    plt.title(
        "Scores of the trained perceptron on the test set over the range of bias weights")
    plt.legend()
    plt.show()

    # plot the ROC curve for the trained perceptron for each bias weight on the test set
    x = [1 - specificity for specificity in specificities]
    y = recalls
    plt.plot(x, y)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve of the trained perceptron on the test set")
    plt.show()

    # visualize the untraied and trained weights as heatmaps, excluding the bias weight
    weights_untrained_matrix = np.array(weights_untrained[1:]).reshape(28, 28)
    weights_trained_matrix = np.array(weights_trained[1:]).reshape(28, 28)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(weights_untrained_matrix, cmap="hot")
    axs[0].set_title("Untrained weights")
    axs[1].imshow(weights_trained_matrix, cmap="hot")
    axs[1].set_title("Trained weights")
    plt.show()

    # get the output of the trained perceptron on the challenge set
    output_challenge_set = []
    weights = weights_trained.copy()
    classified_as_one = {
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }
    classified_as_zero = {
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }
    for i in range(len(challenge_set)):
        input = challenge_set[i]
        input_arr = np.array([1] + input.split("\t"), dtype=float)
        net_input = calculate_net_input(weights, input_arr)
        output = 1 if net_input >= 0 else 0
        output_challenge_set.append(output)
        if output == 1:
            classified_as_one[challenge_set_labels[i].strip()] += 1
        else:
            classified_as_zero[challenge_set_labels[i].strip()] += 1
    # plot the classification data on a table
    fig, ax = plt.subplots()
    ax.axis("off")
    table_data = [["Points"] + [str(i) for i in range(2, 10)]]
    for i in range(2):
        table_data.append([f"as {str(i)}"] + [str((classified_as_zero if i == 0 else classified_as_one)[str(j)])
                          for j in range(2, 10)])
    ax.table(cellText=table_data, loc="center")
    plt.show()
