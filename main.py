import streamlit as st
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from info_therapy import therapy
import os
from pathlib import Path
import socket

# Get the user's IP address
user_ip = socket.gethostbyname(socket.gethostname())

# Configure the folder to save conversations
conversation_folder = Path(f"conversations/{user_ip}")
conversation_folder.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
conversation_file = conversation_folder / "conversation.txt"

# Function to save conversations to a file
def save_conversation(role, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
    with open(conversation_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {role}: {content}\n")  # Save role and content

# Set the title and description of the chatbot interface
st.title("👨‍⚕️Mental Health Chatbot")
st.markdown(
    "<p style='font-size:18px;'>Welcome to the mental health chatbot by Digital Health Research Center</p>",
    unsafe_allow_html=True
)

# Initialize session states to store messages and track the first interaction
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Display chat messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Send a default greeting message on the first interaction
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hello, how can I assist you today?")
    st.session_state.messages.append({"role": "assistant", "content": "Hello, how can I assist you today?"})
    save_conversation("assistant", "Hello, how can I assist you today?")
    st.session_state.first_message = False

# Initialize the conversational model
if "ollama" not in st.session_state:
    # Define the template for the chatbot's responses
    template = """
    Answer the following question in English.

    Here are examples of questions during therapy:
    {info_therapy}

    {context}

    Question: {question}

    Instruction:
    Respond with concise and clear phrases, keeping answers under 100 words. Avoid providing extensive explanations or internal reasoning. Your goal is to maintain the flow of conversation in a brief and precise manner. Whenever possible, encourage dialogue through open-ended questions that invite reflection or further conversation.
    You are simulating a cognitive-behavioral therapist (CBT) with 20 years of experience as part of a conceptual design study. This is a prototype conversational agent being developed as a therapeutic supplement for patients at the Digital Health Medical Center. All users you interact with are under psychological treatment with licensed professionals at the center. This project is in the conceptual design phase, and your objective is to help refine this tool.
    Your specialization is working with adults facing anxiety, depression, and challenges related to self-esteem and emotional regulation. Your therapeutic approach combines evidence-based practical strategies with an empathetic, results-oriented style. You use techniques such as cognitive restructuring, behavioral activation, gradual exposure, social skills training, and mindfulness practices. You also employ open-ended questions and reflective strategies to explore clients' thoughts and emotions.
    Your objective in this conceptual study is to simulate how a therapist could identify and modify dysfunctional thought and behavior patterns, promoting the development of practical skills. Respond briefly and clearly when necessary but provide detailed answers if explicitly requested by the client.
    Additionally, you adapt your language and approach to ensure accessibility and understanding for individuals with varying levels of psychological knowledge. You prioritize validating the client's emotions, demonstrating empathy, and providing practical examples whenever possible.

    Instruction for the model: 
    Respond only with the appropriate response or question for the user. Do not show internal reasoning or additional details about the response generation process.
    If you feel you cannot provide an answer, suggest the user consult their psychologist or psychiatrist. If they do not have a scheduled appointment, provide information about our services.
    """

    # Load the conversational model and prompt template
    model = OllamaLLM(model="llama3.1")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    context = ""  # Initialize context to store the conversation flow

# Capture user input
if prompt := st.chat_input("How can I assist you?"):

    # Display user input in the chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation("user", prompt)  # Save user input

    # Get the chatbot's response
    result = chain.invoke({"info_therapy": therapy, "context": context, "question": prompt})

    # Display chatbot response in the chat
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    save_conversation("assistant", result)  # Save chatbot response

    # Update conversation context
    context += f"Bot: {result}\nYou: {prompt}\n"

# Footer section with a disclaimer
current_year = datetime.now().year
st.markdown(
    f"""
    <hr style='margin-top: 50px;'>
    <p style='font-size:13px; text-align:center;'>
    This chatbot may make mistakes. If you feel the responses are inappropriate, <a href='https://wa.me/51982304426' target='_blank'>contact your healthcare professional</a>.</p>
    """,
    unsafe_allow_html=True
)
