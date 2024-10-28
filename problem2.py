from utills import *
from os import makedirs
import matplotlib.pyplot as plt

learning_rate = 0.01
epochs = 15

# make the necessary directories
makedirs("output", exist_ok=True)
[makedirs(f"output/perceptron_{i}", exist_ok=True) for i in range(10)]

generate_data_sets(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

# train the perceptron for recognizing 9s
simulate_perceptron("9", learning_rate, epochs)
simulate_perceptron("8", learning_rate, epochs)
simulate_perceptron("7", learning_rate, epochs)
simulate_perceptron("6", learning_rate, epochs)
simulate_perceptron("5", learning_rate, epochs)
simulate_perceptron("4", learning_rate, epochs)
simulate_perceptron("3", learning_rate, epochs)
simulate_perceptron("2", learning_rate, epochs)
simulate_perceptron("1", learning_rate, epochs)
simulate_perceptron("0", learning_rate, epochs)

# open the metrics files
[f_untrained_metrics_0, f_untrained_metrics_1, f_untrained_metrics_2, f_untrained_metrics_3, f_untrained_metrics_4, f_untrained_metrics_5, f_untrained_metrics_6,
    f_untrained_metrics_7, f_untrained_metrics_8, f_untrained_metrics_9] = [open(f"output/perceptron_{i}/untrained_metrics.txt", "r") for i in range(10)]
[f_trained_metrics_0, f_trained_metrics_1, f_trained_metrics_2, f_trained_metrics_3, f_trained_metrics_4, f_trained_metrics_5, f_trained_metrics_6,
    f_trained_metrics_7, f_trained_metrics_8, f_trained_metrics_9] = [open(f"output/perceptron_{i}/trained_metrics.txt", "r") for i in range(10)]
# read the metrics files
[untrained_metrics_0, untrained_metrics_1, untrained_metrics_2, untrained_metrics_3, untrained_metrics_4, untrained_metrics_5, untrained_metrics_6,
    untrained_metrics_7, untrained_metrics_8, untrained_metrics_9] = [f.readlines() for f in [f_untrained_metrics_0, f_untrained_metrics_1, f_untrained_metrics_2, f_untrained_metrics_3, f_untrained_metrics_4, f_untrained_metrics_5, f_untrained_metrics_6,
                                                                                              f_untrained_metrics_7, f_untrained_metrics_8, f_untrained_metrics_9]]
[trained_metrics_0, trained_metrics_1, trained_metrics_2, trained_metrics_3, trained_metrics_4, trained_metrics_5, trained_metrics_6, trained_metrics_7, trained_metrics_8, trained_metrics_9] = [f.readlines(
) for f in [f_trained_metrics_0, f_trained_metrics_1, f_trained_metrics_2, f_trained_metrics_3, f_trained_metrics_4, f_trained_metrics_5, f_trained_metrics_6, f_trained_metrics_7, f_trained_metrics_8, f_trained_metrics_9]]
# close the metrics files
[f.close() for f in [f_untrained_metrics_0, f_untrained_metrics_1, f_untrained_metrics_2, f_untrained_metrics_3, f_untrained_metrics_4, f_untrained_metrics_5, f_untrained_metrics_6,
                     f_untrained_metrics_7, f_untrained_metrics_8, f_untrained_metrics_9]]
[f.close() for f in [f_trained_metrics_0, f_trained_metrics_1, f_trained_metrics_2, f_trained_metrics_3, f_trained_metrics_4,
                     f_trained_metrics_5, f_trained_metrics_6, f_trained_metrics_7, f_trained_metrics_8, f_trained_metrics_9]]

# extract individual metrics
error_fractions_untrained = [float(untrained_metrics[0].split(":")[1]) for untrained_metrics in [
    untrained_metrics_0, untrained_metrics_1, untrained_metrics_2, untrained_metrics_3, untrained_metrics_4, untrained_metrics_5, untrained_metrics_6, untrained_metrics_7, untrained_metrics_8, untrained_metrics_9]]
error_fractions_trained = [float(trained_metrics[0].split(":")[1]) for trained_metrics in [
    trained_metrics_0, trained_metrics_1, trained_metrics_2, trained_metrics_3, trained_metrics_4, trained_metrics_5, trained_metrics_6, trained_metrics_7, trained_metrics_8, trained_metrics_9]]
precisions_untrained = [float(untrained_metrics[1].split(":")[1]) for untrained_metrics in [
    untrained_metrics_0, untrained_metrics_1, untrained_metrics_2, untrained_metrics_3, untrained_metrics_4, untrained_metrics_5, untrained_metrics_6, untrained_metrics_7, untrained_metrics_8, untrained_metrics_9]]
precisions_trained = [float(trained_metrics[1].split(":")[1]) for trained_metrics in [
    trained_metrics_0, trained_metrics_1, trained_metrics_2, trained_metrics_3, trained_metrics_4, trained_metrics_5, trained_metrics_6, trained_metrics_7, trained_metrics_8, trained_metrics_9]]
recalls_untrained = [float(untrained_metrics[2].split(":")[1]) for untrained_metrics in [
    untrained_metrics_0, untrained_metrics_1, untrained_metrics_2, untrained_metrics_3, untrained_metrics_4, untrained_metrics_5, untrained_metrics_6, untrained_metrics_7, untrained_metrics_8, untrained_metrics_9]]
recalls_trained = [float(trained_metrics[2].split(":")[1]) for trained_metrics in [
    trained_metrics_0, trained_metrics_1, trained_metrics_2, trained_metrics_3, trained_metrics_4, trained_metrics_5, trained_metrics_6, trained_metrics_7, trained_metrics_8, trained_metrics_9]]
f1_scores_untrained = [float(untrained_metrics[3].split(":")[1]) for untrained_metrics in [
    untrained_metrics_0, untrained_metrics_1, untrained_metrics_2, untrained_metrics_3, untrained_metrics_4, untrained_metrics_5, untrained_metrics_6, untrained_metrics_7, untrained_metrics_8, untrained_metrics_9]]
f1_scores_trained = [float(trained_metrics[3].split(":")[1]) for trained_metrics in [
    trained_metrics_0, trained_metrics_1, trained_metrics_2, trained_metrics_3, trained_metrics_4, trained_metrics_5, trained_metrics_6, trained_metrics_7, trained_metrics_8, trained_metrics_9]]

# plot each metric as a separate barchart figure, with each bar representing a digit. Cluster the untrained and trained bars together
labels = [str(i) for i in range(10)]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, error_fractions_untrained,
                width, label="Untrained")
rects2 = ax.bar(x + width/2, error_fractions_trained, width, label="Trained")
ax.set_ylabel("Scores")
ax.set_title(
    "Error fraction of the perceptron on the test set before and after training")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, precisions_untrained,
                width, label="Untrained")
rects2 = ax.bar(x + width/2, precisions_trained, width, label="Trained")
ax.set_ylabel("Scores")
ax.set_title(
    "Precision of the perceptron on the test set before and after training")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, recalls_untrained,
                width, label="Untrained")
rects2 = ax.bar(x + width/2, recalls_trained, width, label="Trained")
ax.set_ylabel("Scores")
ax.set_title(
    "Recall of the perceptron on the test set before and after training")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, f1_scores_untrained,
                width, label="Untrained")
rects2 = ax.bar(x + width/2, f1_scores_trained, width, label="Trained")
ax.set_ylabel("Scores")
ax.set_title(
    "F1 score of the perceptron on the test set before and after training")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# open the training error fractions files
[f_error_fractions_0, f_error_fractions_1, f_error_fractions_2, f_error_fractions_3, f_error_fractions_4, f_error_fractions_5, f_error_fractions_6,
    f_error_fractions_7, f_error_fractions_8, f_error_fractions_9] = [open(f"output/perceptron_{i}/training_error_fractions.txt", "r") for i in range(10)]
# read the training error fractions files
[error_fractions_0, error_fractions_1, error_fractions_2, error_fractions_3, error_fractions_4, error_fractions_5, error_fractions_6,
    error_fractions_7, error_fractions_8, error_fractions_9] = [f.readlines() for f in [f_error_fractions_0, f_error_fractions_1, f_error_fractions_2, f_error_fractions_3, f_error_fractions_4, f_error_fractions_5, f_error_fractions_6,
                                                                                        f_error_fractions_7, f_error_fractions_8, f_error_fractions_9]]
# close the training error fractions files
[f.close() for f in [f_error_fractions_0, f_error_fractions_1, f_error_fractions_2, f_error_fractions_3, f_error_fractions_4, f_error_fractions_5, f_error_fractions_6,
                     f_error_fractions_7, f_error_fractions_8, f_error_fractions_9]]
# convert the error fractions to floats
error_fractions_0 = [float(fraction) for fraction in error_fractions_0]
error_fractions_1 = [float(fraction) for fraction in error_fractions_1]
error_fractions_2 = [float(fraction) for fraction in error_fractions_2]
error_fractions_3 = [float(fraction) for fraction in error_fractions_3]
error_fractions_4 = [float(fraction) for fraction in error_fractions_4]
error_fractions_5 = [float(fraction) for fraction in error_fractions_5]
error_fractions_6 = [float(fraction) for fraction in error_fractions_6]
error_fractions_7 = [float(fraction) for fraction in error_fractions_7]
error_fractions_8 = [float(fraction) for fraction in error_fractions_8]
error_fractions_9 = [float(fraction) for fraction in error_fractions_9]

# plot the error fractions of the perceptron over the training epochs for all digits on the same plot
plt.plot(range(1, len(error_fractions_0) + 1), error_fractions_0, label="0")
plt.plot(range(1, len(error_fractions_1) + 1), error_fractions_1, label="1")
plt.plot(range(1, len(error_fractions_2) + 1), error_fractions_2, label="2")
plt.plot(range(1, len(error_fractions_3) + 1), error_fractions_3, label="3")
plt.plot(range(1, len(error_fractions_4) + 1), error_fractions_4, label="4")
plt.plot(range(1, len(error_fractions_5) + 1), error_fractions_5, label="5")
plt.plot(range(1, len(error_fractions_6) + 1), error_fractions_6, label="6")
plt.plot(range(1, len(error_fractions_7) + 1), error_fractions_7, label="7")
plt.plot(range(1, len(error_fractions_8) + 1), error_fractions_8, label="8")
plt.plot(range(1, len(error_fractions_9) + 1), error_fractions_9, label="9")
plt.xlabel("Epoch")
plt.ylabel("Error fraction")
plt.title("Error fraction over epochs")
plt.legend()
plt.show()
