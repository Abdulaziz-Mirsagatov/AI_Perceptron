import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, seed, randint


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


def generate_data_sets():
    training_set = []
    training_set_labels = []

    test_set = []
    test_set_labels = []

    challenge_set = []
    challenge_set_labels = []

    # keeps track of the number of images collected for each digit
    count = {
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

        # collect 400 images of each of the digits 0 and 1 for the training set, the remaining 0 and 1 digits into test set, and 100 images of each 2-9 digits for the challenge set
        for i in range(len(labels)):
            label = labels[i].strip()
            # training set
            if (label == "0" or label == "1") and count[label] < 400:
                training_set.append(images[i])
                training_set_labels.append(labels[i])
                count[label] += 1
            # test set
            elif (label == "0" or label == "1"):
                test_set.append(images[i])
                test_set_labels.append(labels[i])
            # challenge set
            elif count[label] < 100:
                challenge_set.append(images[i])
                challenge_set_labels.append(labels[i])
                count[label] += 1

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
