
import tkinter as tk
from tkinter import filedialog

import cv2
import joblib

# Import your LBP class
from lbp.lbp import LocalBinaryPatterns


def classify_roi_on_image(image, model, desc):
    """
    Selects an ROI on a given image, classifies it using the provided model,
    and displays the result. Returns True if an ROI was successfully chosen,
    otherwise False.
    """
    # Let the user select an ROI
    r = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    if r[2] == 0 or r[3] == 0:
        # No ROI selected (canceled)
        return False

    x, y, w, h = r
    roi = image[y:y+h, x:x+w]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = desc.compute(gray_roi)

    prediction = model.predict(hist.reshape(1, -1))[0]
    print(f"[RESULT] Predicted label for selected ROI: {prediction}")

    # Draw the ROI rectangle and label on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        prediction,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Display the image with ROI and prediction
    window_title = f"Classification Result: {prediction}"
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


def main():
    # ------------------------------------------------------------------
    # 1. Load your trained (saved) model
    # ------------------------------------------------------------------
    model_path = "lbp_svc_model.pkl"
    print(f"[INFO] Loading model from {model_path}")
    model = joblib.load(model_path)

    # Create an instance of your LBP descriptor (use the same params as training)
    desc = LocalBinaryPatterns(numPoints=16, radius=2)

    # Create a hidden Tkinter root window for file dialog
    root = tk.Tk()
    root.withdraw()

    while True:
        # ------------------------------------------------------------------
        # 2. Ask user to select an image file
        # ------------------------------------------------------------------
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )

        if not image_path:
            print("[INFO] No file selected. Exiting...")
            break  # Exit the outer loop, ending the program

        # ------------------------------------------------------------------
        # 3. Read the selected image
        # ------------------------------------------------------------------
        image = cv2.imread(image_path)
        if image is None:
            print("[ERROR] Could not load the image. Check file type or path.")
            continue  # Go back to picking a new image

        # Optional: resize the image for convenience (uncomment if desired)
        # image = cv2.resize(image, (800, 600))

        # ------------------------------------------------------------------
        # 4. Classify multiple ROIs on the same image if desired
        # ------------------------------------------------------------------
        while True:
            print("[INFO] Select ROI on the image window.")
            roi_chosen = classify_roi_on_image(image.copy(), model, desc)
            # Note: use a copy if you don't want the drawn rectangles to accumulate.
            # If you *do* want them to accumulate, pass 'image' directly instead of image.copy().

            if not roi_chosen:
                print("[INFO] No ROI selected or canceled. Returning to main menu.")
                break

            # ------------------------------------------------------------------
            # 5. Prompt user: classify another ROI on the same image, or new image, or exit?
            # ------------------------------------------------------------------
            user_input = input(
                "\nEnter:\n"
                "  [S] to select another ROI on the SAME image,\n"
                "  [N] to choose a NEW image,\n"
                "  [E] to EXIT completely.\n"
                "Choice: "
            ).strip().lower()

            if user_input == 's':
                # Continue the loop to select another ROI on the same image
                continue
            elif user_input == 'n':
                # Break the inner loop to select a new image
                break
            elif user_input == 'e':
                # Exit the entire application
                print("[INFO] Exiting application.")
                return
            else:
                print("[INFO] Invalid input. Going back to main menu.")
                break


if __name__ == "__main__":
    main()
