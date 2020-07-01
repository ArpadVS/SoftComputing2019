import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import time
import matplotlib
import matplotlib.pyplot as plt

IMAGE_SIZE = 520


# load single image in RGB
def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


# resizing images to the same resolution
def resize_image(img):
    w, h = IMAGE_SIZE, IMAGE_SIZE
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


# loading all images for train and test
def load_data():
    train_csv = pd.read_csv("train/train_labels.csv")
    data_train = []
    labels_train = []

    # loading train images and labels
    for i in range(len(train_csv)):
        picture = resize_image(load_image("train/" + train_csv["file"][i]))
        # plt.imshow(picture, 'gray')
        # picture = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        data_train.append(picture)
        labels_train.append(train_csv["labels"][i])

    print("Loaded " + str(len(labels_train)) + " images for training.")

    test_csv = pd.read_csv("test/test_labels.csv")
    data_test = []
    labels_test = []

    # loading test images and labels
    for i in range(len(test_csv)):
        picture = resize_image(load_image("test/" + test_csv["file"][i]))
        # picture = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # plt.imshow(picture, 'gray')

        data_test.append(picture)
        labels_test.append(test_csv["labels"][i])

    print("Loaded " + str(len(labels_test)) + " images for testing.")

    return data_train, labels_train, data_test, labels_test


# reshape date from hog to format ideal for SVM classifier
# [[0.00061088]
#  [0.00039584]]  TO  [0.00061088 0.00039584]
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def extract_features(hog_descriptor, images, labels):
    features = []
    for image in images:
        features.append(hog_descriptor.compute(image))

    # print(features[0])
    x = reshape_data(np.array(features))
    # print(x[0])
    y = np.array(labels)

    return x, y


if __name__ == '__main__':
    print("Program started...")
    train_images, train_labels, test_images, test_labels = load_data()

    nbins = 20
    cell_size = (24, 24)
    block_size = (6, 6)

    hog = cv2.HOGDescriptor(_winSize=(IMAGE_SIZE // cell_size[1] * cell_size[1],
                                      IMAGE_SIZE // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    print("Computing HOG for train images...")
    x_train, y_train = extract_features(hog, train_images, train_labels)

    clf_svm = SVC(kernel='linear')

    start = time.time()
    print("Training SVM classifier...")
    clf_svm = clf_svm.fit(x_train, y_train)
    end = time.time()
    print("Training finished in " + str(end - start) + " seconds")

    print("Predicting on train...")
    train_prediction = clf_svm.predict(x_train)

    accuracy = accuracy_score(y_train, train_prediction)
    print("Prediction accuracy on train: ", accuracy, " (", (accuracy * 100), "%)")

    print("Computing HOG for test images...")
    x_test, y_test = extract_features(hog, test_images, test_labels)
    print("Prediction on test...")
    test_prediction = clf_svm.predict(x_test)

    print("\nFirst 10 predicted values for test:")
    print(test_prediction[0:10])
    print("Real values:")
    print(y_test[0:10])
    print()

    accuracy = accuracy_score(y_test, test_prediction)
    print("Prediction accuracy: ", accuracy, " (", (accuracy * 100), "%)")


