import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from skimage.filters import gabor


def select_image():
    file_path = filedialog.askopenfilename(title='Select SEM Image', filetypes=[('Image Files', '*.jpg *.png *.tif *.bmp')])
    return file_path

def get_scale_bar(image):
    messagebox.showinfo("Scale Bar Selection", "Select the scale bar region and press ENTER.")
    roi = cv2.selectROI("Select Scale Bar", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    return x, y, w, h

def get_roi(image):
    messagebox.showinfo("ROI Selection", "Select the region of interest and press ENTER.")
    roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # Extract the cropped area based on ROI
    x, y, w, h = roi
    cropped_image = image[y:y+h, x:x+w]

    # Convert to grayscale
    gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Enhance Contrast Using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray_cropped)

    return contrast_enhanced

def measure_scale_bar(scale_bar_region):
    gray = cv2.cvtColor(scale_bar_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bar_width = np.sum(binary > 0, axis=1).max()
    real_length = float(input("Enter the real length of the scale bar in micrometers (µm): "))
    return real_length / bar_width

def find_best_theta(image, frequency=0.1, theta_range=np.linspace(0, np.pi, 180)):
    responses = []
    for theta in theta_range:
        filtered_image, _ = gabor(image, frequency=frequency, theta=theta)
        responses.append(np.mean(filtered_image))
    best_theta = theta_range[np.argmax(responses)]
    return best_theta

def count_lines_and_distances(filtered_image, theta, pixel_to_micron):
    angle_degrees = np.rad2deg(theta)
    rotated_image = rotate(filtered_image, angle_degrees, reshape=True, order=1, mode='constant', cval=128)
    normalized_image = cv2.normalize(rotated_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_image = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    line_profile = np.sum(binary_image, axis=0)
    height = 0.05 * np.max(line_profile) + np.min(line_profile)
    peaks, _ = find_peaks(line_profile, height=height, distance=5)
    line_distances = np.diff(peaks) * pixel_to_micron
    return len(peaks), line_distances, rotated_image, line_profile, peaks

def main():
    while True:
        file_path = select_image()
        if not file_path:
            break
        image = cv2.imread(file_path)
        scale_x, scale_y, scale_w, scale_h = get_scale_bar(image)
        scale_bar_region = image[scale_y:scale_y+scale_h, scale_x:scale_x+scale_w]
        pixel_to_micron = measure_scale_bar(scale_bar_region)
        gray_roi = get_roi(image)
        best_theta = find_best_theta(gray_roi)
        filtered_image, _ = gabor(gray_roi, frequency=0.1, theta=best_theta)
        num_lines, line_distances, rotated_image, line_profile, peaks = count_lines_and_distances(filtered_image, best_theta, pixel_to_micron)
        print(f"Best theta: {best_theta:.2f} radians")
        print(f"Number of lines: {num_lines}")
        print(f"Distances between lines (µm): {line_distances}")
        plt.figure(figsize=(10, 5))
        plt.imshow(rotated_image, cmap='gray')
        for peak in peaks:
            plt.axvline(peak, color='red', linestyle='--', linewidth=0.5)
        plt.title("Rotated Image with Detected Lines")
        plt.show()
        plt.figure(figsize=(10, 4))
        plt.plot(line_profile)
        plt.scatter(peaks, [line_profile[p] for p in peaks], color='red')
        plt.title("Line Profile with Peaks")
        plt.show()
        retry = messagebox.askyesno("Retry", "Do you want to select another image or ROI?")
        if not retry:
            break

if __name__ == "__main__":
    main()
