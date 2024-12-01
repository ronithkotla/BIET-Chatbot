import os
import json
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.title("Bharat Institute of Engineering & Technology -Chatbot (BIET-bot)")

    # Initialize session state variables if not already present
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


    # Create the radio button options in the sidebar
    choice = st.sidebar.radio("Menu", options=["Chatbot", "Conversation History", "About"])

    # Home Menu
    if choice == "Chatbot":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        user_input = st.chat_input("You:")

        if user_input:
            # Get chatbot response
            response = chatbot(user_input)

            # Store user input and chatbot response in session_state
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": user_input
            })
            st.session_state.conversation_history.append({
                "role": "chatbot", 
                "content": response
            })

            # Display conversation in chat-like format
            for conversation in st.session_state.conversation_history:
                if conversation['role'] == 'user':
                    st.markdown(f"<div style='text-align: right; background-color: #414A4C; padding: 10px; border-radius: 10px;margin: 5px 0;'><strong>You:   </strong>{conversation['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: left; background-color: #000000; padding: 10px; border-radius: 10px;margin: 5px 0;'><strong>Chatbot:   </strong>{conversation['content']}</div>", unsafe_allow_html=True)

            # Clear the input for the next question
            

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        if st.session_state.conversation_history:
            # Display stored conversation history
            for conversation in st.session_state.conversation_history:
                if conversation['role'] == 'user':
                    st.text(f"You: {conversation['content']}")
                else:
                    st.text(f"Chatbot: {conversation['content']}")
                st.markdown("---")
        else:
            st.write("No conversation history yet.")

    elif choice == "About":
        image_path = os.path.abspath("./Infrastructure9.jpg")
        st.image(image_path)
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)
        st.subheader("Note")
        st.write("This chatbot is specifically built for the BIET college so it will answer questions related to college.")
        

if __name__ == '__main__':
    main()
