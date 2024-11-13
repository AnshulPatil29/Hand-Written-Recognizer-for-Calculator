import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import pickle
import torch
from sklearn.preprocessing import LabelEncoder
import cv2
import torch.nn as nn
import torch.nn.functional as F
from skimage import exposure

st.title("Handwritten equation predictor")


st.markdown("""
* Write an equation and predict.            
""")


stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
stroke_color = st.sidebar.color_picker("Stroke color: ", "#000000")
background_color = st.sidebar.color_picker("Background color: ", "#FFFFFF")

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"),
)


if st.sidebar.button("Eraser"):
    stroke_color = background_color
    stroke_width = 25

canvas_height = st.sidebar.slider("Canvas height: ", 100, 800, 400)
canvas_width = st.sidebar.slider("Canvas width: ", 100, 800, 600)

toggle_draw = st.checkbox("Click to toggle drawing on/off")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=background_color,
    height=canvas_height,
    width=canvas_width,
    drawing_mode=drawing_mode,
    key="canvas",
    display_toolbar=toggle_draw
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def object_loader(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

class RegularizedCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(RegularizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)  
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.batch_norm1 = nn.BatchNorm2d(16)  
        self.batch_norm2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 


def load_model():
    model_path = 'reg_cnn_model.pth'
    label_encoder_path = 'LabelEncoder.pkl'
    model = RegularizedCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()
    label_encoder = object_loader(label_encoder_path)
    return model, label_encoder

def threshold_image(image: np.ndarray):
    return (image > 220).astype('uint8') * 255

def preprocess_image_full(image:np.ndarray):
    img_pil = image.convert('L')
    img_array = np.array(img_pil)
    img_equalized = exposure.equalize_adapthist(img_array, clip_limit=0.03)
    img_equalized = (img_equalized * 255).astype('uint8')
    img_thresholded = threshold_image(img_equalized)
    return img_thresholded

def image_to_tensor(img_array):
    
    img_normalized = img_array.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    return img_tensor

def extract_mser_segments(image, resize_shape=(128, 128), padding_factor=2.2, extra_padding=50,show_plots:bool=False): # 2.2 50
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(image)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bounding_boxes = []
    
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        if w > 5 and h > 5:
            bounding_boxes.append((x, y, w, h))

    def merge_bounding_boxes(boxes, overlap_thresh=0.3, vertical_merge_thresh=20):
        merged_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            x1, y1, w1, h1 = boxes[i]
            merged_box = [x1, y1, x1 + w1, y1 + h1]

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                x2, y2, w2, h2 = boxes[j]

                if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2) or \
                   (abs(x1 - x2) < 10 and abs(w1 - w2) < 10 and abs(y1 + h1 - y2) < vertical_merge_thresh):
                    merged_box[0] = min(merged_box[0], x2)
                    merged_box[1] = min(merged_box[1], y2)
                    merged_box[2] = max(merged_box[2], x2 + w2)
                    merged_box[3] = max(merged_box[3], y2 + h2)
                    used[j] = True

            merged_boxes.append((merged_box[0], merged_box[1], merged_box[2] - merged_box[0], merged_box[3] - merged_box[1]))
            used[i] = True

        return merged_boxes

    merged_boxes = merge_bounding_boxes(bounding_boxes)
    merged_boxes = sorted(merged_boxes, key=lambda b: b[0])
    resized_segments = []

    for index, (x, y, w, h) in enumerate(merged_boxes, start=1):
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        segment = image[y:y+h, x:x+w]
        height, width = segment.shape
        
        max_dim = int(max(height, width) * padding_factor)
        padded_segment = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        
        if height > width:
            new_width = int((max_dim / height) * width)
            resized_segment = cv2.resize(segment, (new_width, max_dim))
            x_offset = (max_dim - new_width) // 2
            padded_segment[:, x_offset:x_offset + new_width] = resized_segment
        else:
            new_height = int((max_dim / width) * height)
            resized_segment = cv2.resize(segment, (max_dim, new_height))
            y_offset = (max_dim - new_height) // 2
            padded_segment[y_offset:y_offset + new_height, :] = resized_segment    
        
        extra_padded_segment = np.full((max_dim + 2 * extra_padding, max_dim + 2 * extra_padding), 255, dtype=np.uint8)
        extra_padded_segment[extra_padding:max_dim + extra_padding, extra_padding:max_dim + extra_padding] = padded_segment
    
        final_segment = cv2.resize(extra_padded_segment, resize_shape)
        resized_segments.append(final_segment)
    return resized_segments


def model_predict(model: RegularizedCNN, image: torch.Tensor, label_encoder: LabelEncoder):
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0]

def reconstruct_equation(label_list:list[str]):
    return "".join([label for label in label_list])

INT_MAX = float('inf')  

def precedence(op):
    if op in ('+', '-'):
        return 1
    if op in ('*', '/'):
        return 2
    return 0

def apply_operator(operands, operator):
    if len(operands) < 2:  
        return INT_MAX
    b = operands.pop()
    a = operands.pop()
    if operator == '+':
        operands.append(a + b)
    elif operator == '-':
        operands.append(a - b)
    elif operator == '*':
        operands.append(a * b)
    elif operator == '/':
        if b == 0:
            return INT_MAX  
        operands.append(a / b)

def infix_to_postfix(expression):
    operators = []
    postfix = []
    i = 0
    while i < len(expression):
        if expression[i].isdigit():
            num = ""
            while i < len(expression) and expression[i].isdigit():
                num += expression[i]
                i += 1
            postfix.append(num)
            i -= 1
        elif expression[i] == '(':
            operators.append(expression[i])
        elif expression[i] == ')':
            while operators and operators[-1] != '(':
                postfix.append(operators.pop())
            if not operators:
                return INT_MAX
            operators.pop()
        elif expression[i] in ('+', '-', '*', '/'):
            while operators and precedence(operators[-1]) >= precedence(expression[i]):
                postfix.append(operators.pop())
            operators.append(expression[i])
        else:
            return INT_MAX  
        i += 1
    while operators:
        if operators[-1] == '(':
            return INT_MAX  
        postfix.append(operators.pop())
    return postfix

def evaluate_postfix(postfix):
    if postfix == INT_MAX:
        return INT_MAX
    operands = []
    for char in postfix:
        if char.isdigit() or (char[0] == '-' and len(char) > 1):
            operands.append(int(char))
        else:
            result = apply_operator(operands, char)
            if result == INT_MAX:
                return INT_MAX
    return operands[-1] if operands else INT_MAX

def evaluate_infix(expression):
    if expression[-1] == '=':
        expression = expression[:-1]
    elif expression[-2:] == "--":
        expression = expression[:-2]
    postfix = infix_to_postfix(expression)
    result = evaluate_postfix(postfix)
    return result 


def pipeline(img, model, label_encoder, device, handwritten=False, show_plots=False):
    processed_img = preprocess_image_full(img)
    segments = extract_mser_segments(processed_img, show_plots=show_plots)
    segment_tensors = [image_to_tensor(segment).to(device) for segment in segments]  # Move tensors to device
    labelled_list = [model_predict(model, segmented_image, label_encoder) for segmented_image in segment_tensors]
    equation = reconstruct_equation(labelled_list)
    result = evaluate_infix(equation)
    print(equation)
    if result == float('inf'):
        result = "Error"
    return result

model, label_encoder = load_model()

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data).astype(np.uint8))
        result=pipeline(img,model,label_encoder,device)
        st.write(f"Predicted Output: {result}")
    else:
        st.warning("No image to Predict!")

