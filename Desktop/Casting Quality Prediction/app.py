import streamlit as st
import time
import numpy as np
import tensorflow
from skimage.metrics import structural_similarity as ssim
from skimage import io, color, transform


import pickle

with st.spinner('Application Loading...'):
    model = tensorflow.keras.models.load_model("best_model.h5")

import streamlit as st
import time





# Title and description
st.title("Casting product Labeler")
#st.markdown("Upload an image and receive see where your product fits.")

# File uploader
uploaded_image = st.file_uploader("Upload an image of your casting", type=["jpg", "png", "jpeg"])
image_path2 = "test1.jpeg"

import cv2

import streamlit as st
import cv2
from skimage.metrics import structural_similarity

# Load the base image
base_image = cv2.imread('test1.jpeg')

# Define the function to calculate image similarity
def compare_images(image1, image2):

    test_image_resized = cv2.resize(test_image, (base_image.shape[1], base_image.shape[0]))
    # Convert images to grayscale
    
    image1_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)

    # Calculate structural similarity index (SSIM)
    ssim_score = structural_similarity(image1_gray, image2_gray)

    return ssim_score




# Image display
if uploaded_image:
    
     # Read the image file
    #test_image = st.image(uploaded_image)

    # Convert the test image to OpenCV format
    test_image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    # Display the uploaded image
    st.image(test_image)


    progress_text = "Checking if image is that of a casting..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()



    # Calculate image similarity score
    similarity_score = compare_images(base_image, test_image)
    print(similarity_score)

    # Check if the similarity score is above a threshold
    if similarity_score > 0.4:
        st.success("Your image is relevant!")

        from PIL import Image
        import numpy as np

        def preprocess_uploaded_image(test_image, img_size=(300, 300)):
            # Assuming test_image is a NumPy array representing an image
            # Convert NumPy array to Pillow image
            img = Image.fromarray(test_image)

            # Ensure the image is in RGB mode
            img = img.convert('RGB')

            # Resize the image
            img = img.resize(img_size)

            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Normalize the pixel values to be between 0 and 1
            img_array = img_array / 255.0

            # Add an extra dimension to the array to make it (1, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)

            return img_array
                
        # Preprocess the image for the model (resize, normalize, etc.)
        preprocessed_image = preprocess_uploaded_image(test_image)

        # # Convert the preprocessed image to a NumPy array
        # preprocessed_image = np.array(preprocessed_image)

        # Make a prediction using the pickled model
        prediction = model.predict(preprocessed_image)
        print(prediction)
           
    else:
        st.error("Your image is not relevant, please upload the image of a casting")
else:
    st.info("Please upload an image to predict.")



# Feedback placeholder
st.subheader("Feedback:")
feedback_text = st.empty()

# Button to trigger feedback generation
if st.button("Generate Feedback"):
    progress_text1 = "Making Prediction..."
    my_bar2 = st.progress(0, text=progress_text1)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar2.progress(percent_complete + 1, text=progress_text1)
    time.sleep(1)
    my_bar2.empty()
    # Read the ima

    # Implement your feedback generation logic here
    if prediction < 0.8:
        feedback = "This casting is Defective"#generate_feedback(uploaded_image)
        #feedback_text.write(feedback)
        st.error(feedback)
    elif prediction > 0.8 :
        feedback = "This casting is Ok"#generate_feedback(uploaded_image)
        #feedback_text.write(feedback)
        st.success(feedback)
        