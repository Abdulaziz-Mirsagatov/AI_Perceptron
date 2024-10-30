import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, seed, randint, uniform


def visualize_data(data_path, row=1):
    # Load the data from the text file
    data = np.loadtxt(data_path)

    # Check if the specified row is within bounds
    if row < 1 or row > data.shape[0]:
        print("Row number out of range.")
        return

    # Select the specified row and reshape it into a 28x28 matrix
    # Subtract 1 to convert to zero-based index
    pixel_matrix = data[row - 1].reshape((28, 28))

    # Correct orientation: rotate 90 degrees clockwise and flip horizontally
    pixel_matrix = np.rot90(pixel_matrix, k=-1)  # Rotate 90 degrees clockwise
    pixel_matrix = np.fliplr(pixel_matrix)       # Flip horizontally

    # Plot the 28x28 matrix
    plt.imshow(pixel_matrix, cmap='gray')
    plt.colorbar()
    plt.title(f"Image {row}")  # Title for the specified image
    plt.show()


def generate_data_sets(digits):
    training_set = []
    training_set_labels = []

    test_set = []
    test_set_labels = []

    challenge_set = []
    challenge_set_labels = []

    # keeps track of the number of each digit in the training set
    training_set_count = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }
    # keeps track of the number of each digit in the test set
    test_set_count = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }
    # keeps track of the number of each digit in the challenge set
    challenge_set_count = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }

    with open("data/MNISTnumImages5000_balanced.txt") as f1, open("data/MNISTnumLabels5000_balanced.txt") as f2:
        images = f1.readlines()
        labels = f2.readlines()

        # collect 400 images of each of the specified digits for the training set, 100 for the test set, and 100 for the challenge set
        for i in range(len(labels)):
            label = labels[i].strip()
            # training set
            if (label in digits) and training_set_count[label] < 400:
                training_set.append(images[i])
                training_set_labels.append(labels[i])
                training_set_count[label] += 1
            # test set
            elif (label in digits) and test_set_count[label] < 100:
                test_set.append(images[i])
                test_set_labels.append(labels[i])
                test_set_count[label] += 1
            # challenge set
            elif (label not in digits) and challenge_set_count[label] < 100:
                challenge_set.append(images[i])
                challenge_set_labels.append(labels[i])
                challenge_set_count[label] += 1

    # shuffle the training set and its labels in the same order
    random_seed = randint(1, 10)
    seed(random_seed)
    shuffle(training_set)
    seed(random_seed)
    shuffle(training_set_labels)

    # write out the sets and their labels to text files
    with open("output/training_set.txt", "w") as f, open("output/training_set_labels.txt", "w") as f2, open("output/test_set.txt", "w") as f3, open("output/test_set_labels.txt", "w") as f4, open("output/challenge_set.txt", "w") as f5, open("output/challenge_set_labels.txt", "w") as f6:
        for image in training_set:
            f.write(image)
        for label in training_set_labels:
            f2.write(label)

        for image in test_set:
            f3.write(image)
        for label in test_set_labels:
            f4.write(label)

        for image in challenge_set:
            f5.write(image)
        for label in challenge_set_labels:
            f6.write(label)


def calculate_net_input(weights, inputs):
    return np.dot(weights, inputs)


def calculate_recall(true_positives, false_negatives):  # fraction of positives identified
    return 1 if (true_positives + false_negatives) == 0 else true_positives / (true_positives + false_negatives)


# fraction of negatives identified
def calculate_specificity(false_positives, true_negatives):
    return 1 if (false_positives + true_negatives) == 0 else true_negatives / (false_positives + true_negatives)


# fraction of identified positives that are correct
def calculate_precision(true_positives, false_positives):
    return 1 if (true_positives + false_positives) == 0 else true_positives / (true_positives + false_positives)


# fraction of identified negatives that are correct
def calculate_negative_predictive_value(true_negatives, false_negatives):
    return 1 if (true_negatives + false_negatives) == 0 else true_negatives / (true_negatives + false_negatives)


def calculate_f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


def calculate_accuracy(true_positives, true_negatives, false_positives, false_negatives, balanced=True):
    if balanced:
        return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    else:
        recall = calculate_recall(true_positives, false_negatives)
        specificity = calculate_specificity(false_positives, true_negatives)
        return (recall + specificity) / 2


def get_metrics(output, labels, digit, balanced=True):
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    for i in range(len(output)):
        label = 1 if labels[i].strip() == digit else 0
        if output[i] == 1 and label == 0:
            false_positives += 1
        elif output[i] == 0 and label == 1:
            false_negatives += 1
        elif output[i] == 1 and label == 1:
            true_positives += 1
        else:
            true_negatives += 1
    error_fraction = 1 - \
        calculate_accuracy(true_positives, true_negatives,
                           false_positives, false_negatives, balanced)
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f1_score = calculate_f1_score(precision, recall)
    specificity = calculate_specificity(false_positives, true_negatives)
    return {"error_fraction": error_fraction, "precision": precision, "recall": recall, "f1_score": f1_score, "specificity": specificity}


def simulate_perceptron(digit, learning_rate, epochs, balanced=True):
    print(f"Perceptron for digit {digit}:\n")
    with open("output/training_set.txt") as f, open("output/training_set_labels.txt") as f2, open("output/test_set.txt") as f3, open("output/test_set_labels.txt") as f4, open("output/challenge_set.txt") as f5, open("output/challenge_set_labels.txt") as f6:
        training_set = f.readlines()
        training_set_labels = f2.readlines()

        test_set = f3.readlines()
        test_set_labels = f4.readlines()

        challenge_set = f5.readlines()

        # initialize weights
        weights = []
        for i in range(len(training_set[0].split("\t")) + 1):
            weights.append(uniform(0, 0.5))

        weights_untrained = weights.copy()
        # write out the initial weights to a text file
        with open(f"output/perceptron_{digit}/initial_weights.txt", "w") as f:
            for weight in weights_untrained:
                f.write(str(weight) + "\n")

        # get the output of the untrained perceptron on the training set
        output_training_set = []
        for input in training_set:
            input_arr = np.array([1] + input.split("\t"), dtype=float)
            net_input = calculate_net_input(weights, input_arr)
            output_training_set.append(1 if net_input >= 0 else 0)

        # calculate the error fraction
        metrics_untrained_training = get_metrics(
            output_training_set, training_set_labels, digit, balanced)
        error_fraction_untrained_training = metrics_untrained_training["error_fraction"]
        print(
            f"Error fraction of the untrained perceptron on the training set: {str(error_fraction_untrained_training)}\n")

        # get the output of the untrained perceptron on the test set
        output_untrained_test_set = []
        for input in test_set:
            input_arr = np.array([1] + input.split("\t"), dtype=float)
            net_input = calculate_net_input(weights, input_arr)
            output_untrained_test_set.append(1 if net_input >= 0 else 0)

        # calculate the error fraction, precision, recall, and F1 score
        metrics_untrained_test = get_metrics(
            output_untrained_test_set, test_set_labels, digit, balanced)
        error_fraction_untrained_test = metrics_untrained_test["error_fraction"]
        precision_untrained_test = metrics_untrained_test["precision"]
        recall_untrained_test = metrics_untrained_test["recall"]
        f1_score_untrained_test = metrics_untrained_test["f1_score"]
        # write out the metrics to a text file
        with open(f"output/perceptron_{digit}/untrained_metrics.txt", "w") as f:
            f.write(f"Error fraction: {str(error_fraction_untrained_test)}\n")
            f.write(f"Precision: {str(precision_untrained_test)}\n")
            f.write(f"Recall: {str(recall_untrained_test)}\n")
            f.write(f"F1 score: {str(f1_score_untrained_test)}\n")

        # train the perceptron
        weights = weights_untrained.copy()
        weights_trained = np.array(weights)
        error_fractions = []
        for i in range(epochs):
            output_training_set = []
            for j in range(len(training_set)):
                input_arr = np.array(
                    [1] + training_set[j].split("\t"), dtype=float)
                net_input = calculate_net_input(weights_trained, input_arr)
                output = 1 if net_input > 0 else 0
                output_training_set.append(output)
                label = 1 if training_set_labels[j].strip() == digit else 0
                error = label - output
                weights_trained += learning_rate * error * input_arr
            # calculate the error fraction of the perceptron on the training set
            metrics = get_metrics(
                output_training_set, training_set_labels, digit, balanced)
            error_fraction = metrics["error_fraction"]
            error_fractions.append(error_fraction)
        # write out the error fractions to a text file and write out the trained weights to a different text file
        with open(f"output/perceptron_{digit}/training_error_fractions.txt", "w") as f, open(f"output/perceptron_{digit}/trained_weights.txt", "w") as f2:
            for error_fraction in error_fractions:
                f.write(str(error_fraction) + "\n")
            for weight in weights_trained:
                f2.write(str(weight) + "\n")

        # get the output of the trained perceptron on the test set
        output_trained_test_set = []
        weights = weights_trained.copy()
        for input in test_set:
            input_arr = np.array([1] + input.split("\t"), dtype=float)
            net_input = calculate_net_input(weights, input_arr)
            output_trained_test_set.append(1 if net_input >= 0 else 0)
        # calculate the error fraction, precision, recall, and F1 score
        metrics_trained_test = get_metrics(
            output_trained_test_set, test_set_labels, digit, balanced)
        error_fraction_trained_test = metrics_trained_test["error_fraction"]
        precision_trained_test = metrics_trained_test["precision"]
        recall_trained_test = metrics_trained_test["recall"]
        f1_score_trained_test = metrics_trained_test["f1_score"]
        # write out the metrics to a text file
        with open(f"output/perceptron_{digit}/trained_metrics.txt", "w") as f:
            f.write(f"Error fraction: {str(error_fraction_trained_test)}\n")
            f.write(f"Precision: {str(precision_trained_test)}\n")
            f.write(f"Recall: {str(recall_trained_test)}\n")
            f.write(f"F1 score: {str(f1_score_trained_test)}\n")

        # create a range of bias weights
        weights = weights_trained.copy()
        trained_bias_weight = weights[0]
        new_bias_weights = np.linspace(
            trained_bias_weight - 10, trained_bias_weight + 10, 20)
        new_bias_weights = np.concatenate(
            (new_bias_weights[:11], [trained_bias_weight], new_bias_weights[11:]))
        # write out the new bias weights to a text file
        with open(f"output/perceptron_{digit}/new_bias_weights.txt", "w") as f:
            for bias_weight in new_bias_weights:
                f.write(str(bias_weight) + "\n")

        # get the output of the trained perceptron on the test set for each new bias weight
        # file for writing out the metrics for new bias weights
        f = open(
            f"output/perceptron_{digit}/new_bias_weights_metrics.txt", "w")
        for bias_weight in new_bias_weights:
            weights[0] = bias_weight
            output_trained_test_set = []
            for input in test_set:
                input_arr = np.array([1] + input.split("\t"), dtype=float)
                net_input = calculate_net_input(weights, input_arr)
                output_trained_test_set.append(1 if net_input >= 0 else 0)
            # calculate the error fraction
            metrics = get_metrics(
                output_trained_test_set, test_set_labels, digit, balanced)
            error_fraction = metrics["error_fraction"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1_score = metrics["f1_score"]
            specificity = metrics["specificity"]
            error_fractions.append(error_fraction)
            # write out the metrics to the file
            f.write(f"Error fraction: {str(error_fraction)}\n")
            f.write(f"Precision: {str(precision)}\n")
            f.write(f"Recall: {str(recall)}\n")
            f.write(f"F1 score: {str(f1_score)}\n")
            f.write(f"Specificity: {str(specificity)}\n")
        f.close()

        # get the output of the trained perceptron on the challenge set
        output_challenge_set = []
        weights = weights_trained.copy()
        for i in range(len(challenge_set)):
            input = challenge_set[i]
            input_arr = np.array([1] + input.split("\t"), dtype=float)
            net_input = calculate_net_input(weights, input_arr)
            output = 1 if net_input >= 0 else 0
            output_challenge_set.append(output)
        # write out the output of the trained perceptron on the challenge set to a text file
        with open(f"output/perceptron_{digit}/challenge_set_output.txt", "w") as f:
            for output in output_challenge_set:
                f.write(str(output) + "\n")
