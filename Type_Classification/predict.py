import argparse
import os

import cv2
import joblib  # For saving and loading models
from imutils import paths
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from lbp.lbp import LocalBinaryPatterns


def main():
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",default='IAI_Aluminum_Data',
                    help="Path to the main dataset folder")
    ap.add_argument("-m", "--model",default='lbp_svc_model.pkl',
                    help="Path to save the trained model (e.g., model.pkl)")
    args = vars(ap.parse_args())

    # Initialize LBP descriptor
    desc = LocalBinaryPatterns(numPoints=16, radius=2)

    data = []
    labels = []

    # Grab all image paths in the dataset
    image_paths = list(paths.list_images(args["dataset"]))

    # Loop over the image paths
    for imagePath in image_paths:
        # Read the image
        image = cv2.imread(imagePath)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Compute LBP histogram
        hist = desc.compute(gray)
        # The label is the name of the subfolder
        label = os.path.basename(os.path.dirname(imagePath))
        
        data.append(hist)
        labels.append(label)

    # Split the data into train (80%) and test (20%)
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train a Linear SVM
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(trainData, trainLabels)

    # Predict on the test set
    predictions = model.predict(testData)
    # Compute the accuracy
    accuracy = accuracy_score(testLabels, predictions)
    print(f"[INFO] Accuracy on the test set: {accuracy:.4f}")

    # Save the model to disk
    joblib.dump(model, args["model"])
    print(f"[INFO] Model saved to {args['model']}")

if __name__ == "__main__":
    main()
