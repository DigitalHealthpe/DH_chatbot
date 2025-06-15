# Mental Health Chatbot by Digital Health Research Center

This repository contains the code for the Mental Health Chatbot, designed to provide mental health assistance using advanced AI technology powered by Llama 3.1. Follow this guide to install and run the chatbot locally.

---

## Requirements

To use this chatbot, you need the following:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8 | Latest 3.x |
| **RAM** | 16 GB | 32 GB |
| **GPU** | Not required | CUDA-compatible for faster inference |
| **Internet** | Needed to download the model |
| **Ollama** | Latest version | – |
| **Git** | Required only if you will `git clone` the repo | – |


---

## Installation Guide

### Step 1: Verify Python & pip

Ollama is required to manage and run the Llama 3.1 model. Download and install it from [Ollama's official website](https://ollama.ai/download).

Once installed, verify it by running the following command in your terminal:

```bash
python --version
pip --version
```
If either command is not recognized, download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/) and be sure to check “Add Python to PATH.”

---

### Step 2: Verify Git


```bash
git --version
```
If Git is missing, install it from [https://git-scm.com/downloads]

---


### Step 3: Install Ollama

Ollama is required to manage and run the Llama 3.1 model. Download and install it from [Ollama's official website](https://ollama.ai/download).

Once installed, verify it by running the following command in your terminal:

```bash
ollama
```
You should see a help menu or version output indicating Ollama is installed correctly.

---

### Step 4: Pull the Llama 3.1 Model

Use the following commands to pull and verify the Llama 3.1 model:

```bash
ollama pull llama3.1
ollama list
```
The `ollama list` command should display `llama3.1` as one of the available models.

---

### Step 5: Install Python Dependencies

Run the following commands to install all necessary Python packages:

```bash
pip install streamlit
pip install langchain
pip install langchain-ollama
pip install ollama
pip install pathlib
```

These dependencies include libraries for managing the AI model, building the chatbot interface, and running it as a web app. Specifically, ensure the following libraries are available in your environment:

- `streamlit`: For creating the chatbot interface.
- `datetime`: For handling timestamps.
- `langchain-ollama`: For integrating Llama 3.1 with LangChain.
- `langchain-core`: For prompts and chaining logic.
- `os`: For handling file and directory operations.
- `pathlib`: For creating and managing directories.
- `socket`: For obtaining user IP.

---

### Step 6: Clone the GitHub Repository

Download the project code from the Digital Health repository:

```bash
git clone https://github.com/DigitalHealthpe/DH_chatbot.git
cd DH_chatbot
```

---

### Step 7: Run the Chatbot

Start the chatbot by running the following command from the project directory:

```bash
streamlit run main.py --server.port 8501
```

This will launch the chatbot on `http://localhost:8501` by default. Open this URL in your browser to interact with the chatbot.

---

## How to Use the Chatbot

1. **Access the chatbot:**
   Open the browser and navigate to `http://localhost:8501`.

2. **Interact with the chatbot:**
   Ask questions or start a conversation, and the chatbot will respond using Llama 3.1's advanced language capabilities.

3. **Error Reporting:**
   If you encounter any inappropriate responses or issues, consult your healthcare professional and report the error.

---

## Screenshots

### Chatbot Interface
![Chatbot Interface](https://research.digitalhealth.pe/wp-content/uploads/2024/12/DH_chatbot.png)


---

## Troubleshooting

- **Issue:** `ollama: command not found`  
  **Solution:** Ensure Ollama is installed correctly and added to your system PATH.

- **Issue:** `llama3.1 not found after pulling`  
  **Solution:** Retry `ollama pull llama3.1` and verify your internet connection.

- **Issue:** `streamlit: command not found`  
  **Solution:** Ensure Python and pip are installed, and reinstall Streamlit using `pip install streamlit`.

---

## Contributions

Feel free to contribute by opening issues or submitting pull requests to improve the chatbot.

---


## Contact

For support or inquiries, contact [Digital Health Research Center](mailto:info@digitalhealth.pe).
