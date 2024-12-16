# Real Time hand written equation recognizing calculator

## Description
This project uses a Regularized CNN built using pytorch and trained on https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset?select=train dataset so that it can segment, recognize and parse equations written on streamlit canvas.

## Features
- Multidigit recognitions
- Equation parser using stack

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [How the model was trained](#Training)
- [Future Work and Shortcomings](#future-work-and-shortcomings)

---

## Installation

### Prerequisites
- Python 3.x
- python libraries listed in requirements.txt
- **Options** CUDA 11.8 was used to train this on a RTX 3060 GPU otherwise pytorch would not detect the GPU

### Steps to Install
1. Clone the repository:
    ```bash
    git clone https://github.com/AnshulPatil29/Hand-Written-Recognizer-for-Calculator.git
    ```
2. Navigate to the repository:
    ```bash
    cd 7_Real-Time-Hand-Written-Recognizer-for-Calculator
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
> Side note: The Ignore folder contains the example image in this readme, that can be deleted.
---

## Usage

### Running the Model
Run the app file
```bash
python app.py
```
Then run the command it gives you, should be this one
```bash
streamlit run app.py
```
This should open the app on your default browser 

---

### Example output
![UI output of app](Ignore/example.png)


## Training

The dataset was augmented to create a total of 10,000 images. This augmentation was necessary because the images fed into the model after segmentation differed from the original training data. The following process was followed to generate the augmented dataset:

1. Random equations were generated.
2. Corresponding images were randomly sampled and concatenated.
3. These images were then segmented and stored, resulting in a dataset of around 10,000 images.

Note that the **division** and **equal-to** symbols were excluded from the dataset because the MSER (Maximally Stable Extremal Regions) segmentation technique failed to properly segment these symbols.

A **regularized CNN model** was trained for 10 epochs to prevent overfitting, which was a challenge faced by the previously used simple CNN model. 

### Results:
- The model achieved **93% training accuracy**.
- The model performed with **100% accuracy** on the validation dataset. However, this high accuracy is likely a byproduct of the test data choice, which included easy-to-recognize symbols, and the exclusion of troublesome symbols.
- Subjective testing revealed that the model worked best when digits had **clear vertical boundaries** between them.

The exclusion of the division and equal-to symbols, along with other specific preprocessing techniques, contributed to these results.

## future-work-and-shortcomings
### Model Shortcomings

As noted during the training process, this model has the following limitations:

1. **Inability to Predict Certain Symbols**:
   - The model cannot predict the division symbol (‘/’) and the decimal symbol (‘.’).
   - Although a fix has been applied for the equal symbol (‘=’), it is important to understand how it functions. During segmentation, the ‘=’ symbol is often misinterpreted as two horizontal lines, leading the model to predict them as two consecutive minus symbols (‘-’). To correct this, if the model identifies two consecutive ‘-’ at the end of an equation, they are assumed to represent an ‘=’.

2. **Limited Training Data**:
   - The model's training data consisted solely of integers, resulting in the exclusion of the decimal symbol by accident. Though, this issue can be fixed fairly easily.

3. **Preprocessing Requirements**:
   - The model requires preprocessing to effectively work on equations written on paper. This is necessary to manage occlusion and shadows that create gradients.
   - Adaptive histogram equalization can be used to address this, followed by applying the minima of the histogram to threshold the image.

### future work
* add support for decimal symbol
* add option to upload image of equation written on plain paper
* handle negative digits during parsing
