from utills import *
from os import makedirs
import matplotlib.pyplot as plt

digit = "1"
learning_rate = 0.01
epochs = 10

# make the necessary directories
makedirs("output", exist_ok=True)
makedirs(f"output/perceptron_{digit}", exist_ok=True)

generate_data_sets(["0", "1"])

# train the perceptron for recognizing 1s
simulate_perceptron(digit, learning_rate, epochs)

with open(f"output/perceptron_{digit}/untrained_metrics.txt", "r") as f1, open(f"output/perceptron_{digit}/training_error_fractions.txt", "r") as f2, open(f"output/perceptron_{digit}/trained_metrics.txt", "r") as f3, open(f"output/perceptron_{digit}/new_bias_weights.txt", "r") as f4, open(f"output/perceptron_{digit}/new_bias_weights_metrics.txt", "r") as f5, open(f"output/perceptron_{digit}/initial_weights.txt", "r") as f6, open(f"output/perceptron_{digit}/trained_weights.txt", "r") as f7, open(f"output/perceptron_{digit}/challenge_set_output.txt", "r") as f8, open("output/challenge_set_labels.txt", "r") as f9:
    untrained_metrics_test = f1.readlines()
    error_fractions_untrained_test = [
        float(fraction) for fraction in f2.readlines()]
    trained_metrics_test = f3.readlines()
    new_bias_weights = [float(weight) for weight in f4.readlines()]
    new_bias_weights_metrics = f5.readlines()
    weights_untrained = [float(weight) for weight in f6.readlines()]
    weights_trained = [float(weight) for weight in f7.readlines()]
    challenge_set_output = [int(output) for output in f8.readlines()]
    challenge_set_labels = f9.readlines()

    # print out the performance metrics of the untrained perceptron on the test set
    print("Untrained Perceptron Metrics")
    for metric in untrained_metrics_test:
        print(metric.strip())

    # plot the error fractions of the perceptron over the training epochs
    plt.plot(range(1, len(error_fractions_untrained_test) + 1),
             error_fractions_untrained_test)
    plt.xlabel("Epoch")
    plt.ylabel("Error fraction")
    plt.title("Error fraction over epochs")
    plt.show()

    # bar plot of the error fraction, precision, recall, and F1 score of the perceptron on the test set before and after training
    ef_ut, precision_ut, recall_ut, f1_ut = [
        float(metric.split(":")[1].strip()) for metric in untrained_metrics_test]
    ef_t, precision_t, recall_t, f1_t = [
        float(metric.split(":")[1].strip()) for metric in trained_metrics_test]
    labels = ["Error fraction", "Precision", "Recall", "F1 score"]
    untrained = [ef_ut, precision_ut, recall_ut, f1_ut]
    trained = [ef_t, precision_t, recall_t, f1_t]
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

    # plot the error fraction, precision, recall, and F1 score over the range of bias weights for the trained perceptron on the test set on the same plot
    metrics = [metric.split(":")[1].strip()
               for metric in new_bias_weights_metrics]
    error_fractions = [float(metrics[i]) for i in range(0, len(metrics), 5)]
    precisions = [float(metrics[i]) for i in range(1, len(metrics), 5)]
    recalls = [float(metrics[i]) for i in range(2, len(metrics), 5)]
    f1_scores = [float(metrics[i]) for i in range(3, len(metrics), 5)]
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
    # plot the ROC curve for the trained perceptron on the test set
    specificities = [float(metrics[i]) for i in range(4, len(metrics), 5)]
    x = [1 - specificity for specificity in specificities]
    y = recalls
    plt.plot(x, y)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve of the trained perceptron on the test set")
    plt.show()

    # visualize the untraied and trained weights as heatmaps, excluding the bias weight
    weights_untrained_matrix = np.array(
        weights_untrained[1:]).reshape(28, 28)
    weights_trained_matrix = np.array(weights_trained[1:]).reshape(28, 28)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(weights_untrained_matrix, cmap="hot")
    axs[0].set_title("Untrained weights")
    axs[1].imshow(weights_trained_matrix, cmap="hot")
    axs[1].set_title("Trained weights")
    plt.show()

    # plot the classification of challenge set data on a table
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
    for i in range(len(challenge_set_output)):
        if challenge_set_output[i] == 1:
            classified_as_one[challenge_set_labels[i].strip()] += 1
        else:
            classified_as_zero[challenge_set_labels[i].strip()] += 1
    fig, ax = plt.subplots()
    ax.axis("off")
    table_data = [["Points"] + [str(i) for i in range(2, 10)]]
    for i in range(2):
        table_data.append([f"as {str(i)}"] + [str((classified_as_zero if i == 0 else classified_as_one)[str(j)])
                                              for j in range(2, 10)])
    ax.table(cellText=table_data, loc="center")
    plt.show()
