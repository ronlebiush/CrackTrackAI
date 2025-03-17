# CrackTrackAI

CrackTrackAI is a project designed to detect and analyze structural cracks using artificial intelligence techniques. 
This project aimed to develop a system that can assist or even replace experts in the fractographic analysis of scanning electron microscope (SEM) images of aircraft parts subjected to full-scale fatigue testing (FSFT).
The project had two main objectives: 
(1) classifying fractures into static or fatigue types and 
(2) counting striation lines in fatigue fracture images. 
Using image processing techniques such as Local Binary Pattern (LBP), Support Vector Classifier (SVC), and filters like the Gabor filter, 
we aimed to automate and enhance these processes. 
The results showed promising accuracy in classifying fractures and effectively counting striation lines, 
providing an automated system that could potentially save time and resources in aircraft testing.

## Features

- **Type_Classification**: Utilizes machine learning models to identify the type of fracture.
- **Line_Detection**: identify the striation lines and calculate the distance between them .
-
## Installation

To set up the CrackTrackAI environment, follow these steps:

1. **Clone the Repository**:
   git clone https://github.com/ronlebiush/CrackTrackAI.git
   cd CrackTrackAI
2. **Create and Activate a Virtual Environment:**
    For Unix-based systems:
      python3 -m venv env
      source env/bin/activate
    For Windows systems:
      python -m venv env
      .\env\Scripts\activate
3. **Install Dependencies:**
    pip install -r requirements.txt

## Usage
# Type_Classification
  to run: 
  use python Type_Classification/gui_lbp_svc.py and choose the ROI you would like to classify
  to extend:
  if you want to add more types of cracks or different metals you can add directory with the name of the class and example tiles in that directory
  you can create tiles from a image using the split_image.py script or the crop_raw_data.py script to manualy select tiles
  then you need to run predict.py to create a new model.pkl file

# Line_Detection
  to run use python Line_Detection/gui_line_detection.py
    Select the SEM image for analysis.
    Mark the scale bar (usually in the bottom-right corner) and press Enter.
    Enter the real length of the scale bar in micrometers (µm).
    Select the ROI, press Enter, and wait for results.
  
