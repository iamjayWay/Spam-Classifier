# Import the Streamlit library for building interactive web applications
import streamlit as st  
# Import joblib for loading and saving pre-trained model pipelines
import joblib  
# Import NumPy for numerical operations such as calculating probabilities
import numpy as np  

# Define a simple text preprocessing function used during unpickling the vectorizer.
# This function converts the input text to lowercase.
def preprocess_text(text):
    return text.lower()  # Convert and return the text in lowercase

# Load the pre-trained model pipeline from the current directory.
# This pipeline includes both the vectorizer and the classifier.
model = joblib.load("test_classifier_pipeline.pkl")
# Load the pre-trained vectorizer from the current directory.
vectorizer = joblib.load("vectorizer.pkl")

# Define a function to predict the message category (spam or legit) using the model.
def predict_spam(message):
    # Transform the input message into a feature vector using the loaded vectorizer.
    transformed_text = vectorizer.transform([message])
    # Use the model to predict the label for the transformed text and extract the first prediction.
    prediction = model.predict(transformed_text)[0]
    # Determine the maximum probability from the predicted probabilities for each class.
    confidence = np.max(model.predict_proba(transformed_text))
    # Return the predicted label and the associated confidence level.
    return prediction, confidence

# Set up the Streamlit page configuration with a custom title.
st.set_page_config(page_title="Email/SMS Spam Classifier")

# Display the title of the web application.
st.title("📩 Email/SMS Spam Classifier")

# Display a markdown header for the input section.
st.markdown("#### Enter an Email or SMS Message Below:")
# Create a text area for user input.
user_input = st.text_area("Type or paste a message here...")

# Initialize session state variables to store messages if they do not already exist.
if "spam_messages" not in st.session_state:
    st.session_state.spam_messages = []  # List for storing messages classified as spam
if "legit_messages" not in st.session_state:
    st.session_state.legit_messages = []  # List for storing messages classified as legitimate

# When the "Predict" button is clicked, process the input.
if st.button("🔍 Predict"):
    # Check if the input is empty or contains only whitespace.
    if not user_input.strip():
        st.warning("⚠️ Please enter some text!")  # Display a warning if no text is entered
    else:
        # Convert the input message to lowercase for uniform processing.
        user_input_lower = user_input.lower()
        # Check if the input contains any unsafe links (using "http://" instead of "https://").
        if "http://" in user_input_lower:
            # Append the message along with its category to the spam_messages list in session state.
            st.session_state.spam_messages.append((user_input, "Unsafe Link Spam"))
            # Display an error message indicating the message is spam due to unsafe links.
            st.error("🚨 This message is identified as spam due to unsafe links (HTTP instead of HTTPS).")
        else:
            # Use the predict_spam function to get the predicted label and confidence for the input message.
            prediction, confidence = predict_spam(user_input)
            # If the prediction is 1 (spam), then append the message to spam_messages and display an error message.
            if prediction == 1:
                st.session_state.spam_messages.append((user_input, "Spam"))
                st.error(f"🚨 This message is classified as **Spam** with {confidence * 100:.2f}% confidence.")
            else:
                # If the prediction is 0 (legit), append the message to legit_messages and display a success message.
                st.session_state.legit_messages.append(user_input)
                st.success(f"✅ This message is classified as **Legit** with {confidence * 100:.2f}% confidence.")

# Provide a checkbox to toggle the view of junk (spam) messages.
if st.checkbox("📂 View Junk Messages"):
    st.markdown("### 🗑 Junk Messages")
    # Loop through each spam message in the session state and display it.
    for msg, category in st.session_state.spam_messages:
        st.warning(f"{category}: {msg}")

# Provide a checkbox to toggle the view of legitimate messages.
if st.checkbox("📂 View Legit Messages"):
    st.markdown("### ✅ Legit Messages")
    # Loop through each legitimate message in the session state and display it.
    for msg in st.session_state.legit_messages:
        st.success(msg)
