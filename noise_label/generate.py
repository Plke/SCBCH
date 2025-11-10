import torch
import numpy as np
import h5py


def add_noise_to_labels(labels, noise_rate):
    num_samples, num_labels = labels.shape
    num_noise = int(num_samples * noise_rate)
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    for i in (noise_indices):
        ones_indices = np.where(labels[i, :] == 1)[0]
        zeros_indices = np.where(labels[i, :] == 0)[0]
        if len(ones_indices) > 0:
            j = np.random.choice(ones_indices)
            labels[i, j]=0

        if len(zeros_indices) > 0:
            j = np.random.choice(zeros_indices)
            labels[i, j]=1
    return labels


def generate_noise_F(noise):
    noise_rate = noise
    data = h5py.File("./data/MIRFlickr.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File(
            "./noise_label/mirflickr25k-lall-noise_{}.h5".format(i), "w"
        )

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()


def generate_noise_N(noise):
    noise_rate = noise
    data = h5py.File("./data/NUS-WIDE.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File("./noise_label/nus-wide-tc21-lall-noise_{}.h5".format(i), "w")

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()


def generate_noise_M(noise):
    noise_rate = noise
    data = h5py.File("./data/MS-COCO.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File("./noise_label/MSCOCO-lall-noise_{}.h5".format(i), "w")

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()


def generate_noise_I(noise):
    noise_rate = noise
    data = h5py.File("./data/IAPR.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File("./noise_label/IAPR-lall-noise_{}.h5".format(i), "w")

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()


if __name__ == "__main__":
    noise_rate = [0.2,0.5,0.8]
    # generate_noise_F(noise_rate)
    # generate_noise_I(noise_rate)
    generate_noise_M(noise_rate)
    # generate_noise_N(noise_rate)
