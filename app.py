from flask import Flask, render_template, request, jsonify
import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever#, create_stuff_documents_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_ollama import ChatOllama

app = Flask(__name__)

# Set your Groq API key
# os.environ["GROQ_API_KEY"] = "api_key"

# Load and process documents
# loader = TextLoader("./cv.txt")
# pages = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = splitter.split_documents(pages)

MODEL = 'llama3.1'
embeddings = OllamaEmbeddings(model=MODEL)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./vector_db_dir",
)

retriever = vectorstore.as_retriever()

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant"
)

# # Use Ollama for the language model
# MODEL = 'llama3.1'
# model = ChatOllama(model=MODEL, temperature=0)

# Set up the conversation chain
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

system_prompt = (
    # "Assume you are Anand Sharma a final year undergrad looking for jobs , answer to the questions based on"
    # "a given context and information about Anand."
    # "Answer the following question using both the provided context and your general knowledge."
    # "If the context is relevant, use it to inform your answer. If not, rely on your general knowledge."
    # "Be concise and go straight to the point."
    
    "Assume you are Anand Sharma a final year undergrad , answer to the questions based on"
    "a given context and information about Anand. "
    "Be as concise as possible and go straight to the point."
    "Answer the question based on the context. If you can't answer the"
    "question, reply - 'I don't know'."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Manage chat history
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    session_id = request.json.get('session_id', 'default_session')
    
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    
    return jsonify({"answer": response["answer"]})

if __name__ == '__main__':
    app.run(debug=True)