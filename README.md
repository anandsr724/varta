# Varta: Personal Chatbot Using Retrieval-Augmented Generation (RAG)

Varta is a conversational chatbot designed to simulate personalized responses using Retrieval-Augmented Generation (RAG). This project utilizes LangChain and Groq APIs, integrated with Flask for a web-based interactive chat experience. Varta leverages customized embeddings to enhance response relevance and simulate a conversational style aligned with personal interactions.

## Features

- Retrieval-Augmented Generation (RAG): Uses RAG for generating contextually accurate answers based on past     conversations and preloaded context.
- Personalized Embeddings: Custom embeddings (using models like Hugging Face’s) encode user queries, enhancing response accuracy.
- Real-time Interaction: Flask-based web interface for immediate user interaction and response generation.
- Session-based History: Maintains chat history per session for continuity in responses.

## Tech Stack

- Backend: Flask for web application and routing.
- NLP Models: LangChain, Groq, and custom embeddings via Ollama and Chroma for vector storage.
- APIs: Groq API for model interaction.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone the repository:

```bash
git clone https://github.com/anandsr724/varta.git
cd varta
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up API keys:
   Add your Groq API key to config.json or set it in your environment variables:

```json
Copy code
{
  "GROQ_API_KEY": "your_api_key_here"
}
```

4. Run the application:

```bash
python app.py
```

5. Access the web app:
   Open your browser and go to http://127.0.0.1:5000 to start interacting with the chatbot.

## Usage

- Web Interface: The Flask server hosts a web page where users can ask questions, and the chatbot responds based on both preloaded information and conversation history.
- Session Management: Each session retains chat history, making responses context-aware and personalized.

## Configuration

- Custom Embeddings: OllamaEmbeddings is used to encode queries, with model parameters set in app.py.
- Session History: Varta's ChatMessageHistory stores each user's session history, which the model references to provide coherent responses.
- Conversation Prompts: Modify the system prompts in app.py for further customization of the chatbot’s conversational style.

### Key Components

- RAG Chain: Combines a history-aware retriever and a question-answering chain.
- Embeddings & Vector Storage: Uses OllamaEmbeddings with Chroma to encode user input and retrieve relevant context.
- Response Chain: The RAG approach generates answers by retrieving and processing relevant context.
