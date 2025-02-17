import streamlit as st
import tensorflow as tf
import numpy as np
import json
import streamlit.components.v1 as components

# Load disease details from JSON file
def load_disease_details():
    with open(r"S:\Plant_Disease_Dataset\Plant_Disease_Dataset\disease_details.json", "r") as file:
        return json.load(file)

disease_details = load_disease_details()

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Recognition", "Know More"]
)

# Main Pages
if app_mode == "Home":
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(
        """
    Welcome to the **Plant Disease Detection System**! 🌿🔍

    This system helps in identifying plant diseases efficiently using advanced machine learning algorithms. 
    All you need to do is upload an image of a plant leaf, and our system will analyze it and detect any potential diseases. 
    Together, we can protect our crops and ensure a healthier harvest!

    ### How It Works:
    1. **Upload Image**: Go to the **Disease Recognition** page and upload an image of a plant leaf.
    2. **Analysis**: The system processes the image using trained machine learning models.
    3. **Results**: The system identifies the disease (if any) and provides treatment options.

    ### Why Choose Our System?
    - **Accuracy**: Powered by a robust machine learning model trained on thousands of plant leaf images.
    - **User-Friendly**: Easy-to-use interface for a seamless experience.
    - **Fast and Efficient**: Instant results that allow for quick action on your crops.

    ### Get Started:
    - Click on **Disease Recognition** from the sidebar, upload a plant image, and let the system work its magic!
    - For more details on the dataset and methodology, head over to the **About** page.

    ### Our Vision:
    We aim to make crop disease identification accessible, empowering farmers and gardeners around the world to take proactive measures and protect their plants.

    ### Get In Touch:
    For further inquiries or collaborations, feel free to reach out to us!
    """
    )

elif app_mode == "About":
    st.header("About the Dataset")
    st.markdown(
        """
        #### Dataset Overview
        The **Plant Disease Dataset** is a collection of RGB images of healthy and diseased plant leaves. The dataset has been categorized into 38 different classes, representing different diseases and healthy plants.

        The dataset is split into training, validation, and test sets:
        - **Train Set**: 70,295 images
        - **Test Set**: 33 images (for model prediction)
        - **Validation Set**: 17,572 images

        #### Dataset Details:
        - The dataset includes leaves from a variety of plants, such as apple, tomato, grape, and more.
        - Each image is labeled as either **healthy** or as a specific disease class.
        - The images are processed and resized to 128x128 pixels for analysis by the trained model.

        #### How the Model Works:
        The model uses Convolutional Neural Networks (CNNs) to classify plant diseases based on leaf image inputs. The training process involved augmenting the dataset to create a robust model capable of identifying diseases with high accuracy.

        #### Challenges:
        - **Data Imbalance**: Some disease classes may have more images than others, leading to potential bias.
        - **Generalization**: While the model performs well on the provided dataset, the ability to generalize to unseen data is always a concern.

        #### Goals:
        - **Quick Identification**: To quickly identify plant diseases, reducing the time between detection and treatment.
        - **Scalable Solution**: The system can be scaled to include more plant types and diseases as more data becomes available.
        """
    )

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if test_image and st.button("Predict"):
        st.snow()
        st.write("Analyzing the image...")
        result_index = model_prediction(test_image)

        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_disease = class_name[result_index]
        st.success(f"The model predicts: **{predicted_disease}**")

        # Save to session state
        st.session_state["predicted_disease"] = predicted_disease

elif app_mode == "Know More":
    st.header("Chat with Us")
    
    # Embed Chatbase using iframe
    chatbase_iframe = """
    <iframe 
        src="https://www.chatbase.co/chatbot-iframe/SsK0OGxPmY4ED8s8HyAJu" 
        width="100%" 
        height="600px" 
        frameborder="0">
    </iframe>
    """
    components.html(chatbase_iframe, height=600, width=800)

    # Custom HTML and JavaScript for Floating Chatbot
    floating_chat_html = """
    <style>
        /* Floating button style */
        #floating-chat-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 15px;
            border-radius: 50%;
            font-size: 24px;
            cursor: pointer;
            z-index: 9999;
        }

        /* Modal content */
        .chat-modal {
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            overflow: auto;
        }

        .chat-modal-content {
            position: absolute;
            bottom: 0;
            width: 100%;
            height: 80%;
            background-color: white;
            border-top-left-radius: 20px;
            border-top-right-radius: 20px;
        }

        .chat-modal iframe {
            width: 100%;
            height: 100%;
        }
    </style>

    <!-- Floating button -->
    <button id="floating-chat-btn" onclick="openChat()">💬</button>

    <!-- Modal to show Chatbase -->
    <div id="chat-modal" class="chat-modal">
        <div class="chat-modal-content">
            <iframe src="https://www.chatbase.co/chatbot-iframe/SsK0OGxPmY4ED8s8HyAJu" frameborder="0"></iframe>
        </div>
    </div>

    <script>
        function openChat() {
            document.getElementById('chat-modal').style.display = "block";
        }

        // Close modal when clicking outside the iframe
        window.onclick = function(event) {
            if (event.target == document.getElementById('chat-modal')) {
                document.getElementById('chat-modal').style.display = "none";
            }
        }
    </script>
    """
    components.html(floating_chat_html, height=600)
