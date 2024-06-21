import streamlit as st
from PIL import Image
import numpy as np
import torch

# Global variable to hold the model
model = None

# Function to load YOLOv5 model
def load_model():
    global model
    if model is None:
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects(model, image):
    # Perform inference
    results = model(image)

    return results.pandas().xyxy[0]

def main():
    st.title('Image Components Identification App')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        # Button to trigger the analysis
        if st.button('Analyse Image'):
            st.write("Analysing...")
            
            # Convert PIL image to numpy array
            img = np.array(image)
            
            # Load model
            model = load_model()
            
            # Perform object detection
            object_results = detect_objects(model, img)
            
            # Display object counts 
            object_counts = object_results['name'].value_counts()
            
            st.write(f"Total Objects Detected: {len(object_results)}")
            
            st.write("Detected objects:")
            for name, count in object_counts.items():
                st.write(f"- {name}: {count} occurrences")

if __name__ == '__main__':
    main()
