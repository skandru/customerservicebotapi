from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import openai
import os


# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = Ollama(model="llama3")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Initialize the index and app
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=1)

app = Flask(__name__)
CORS(app)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    if data is None or 'question' not in data:
        logging.warning('Invalid request: No question provided')
        return jsonify({'error': 'Invalid request. No question provided.'}), 400

    question = data['question']
    logging.info(f'Question received: {question}')
    response = query_engine.query(question)
    serialized_response = str(response)

    return jsonify({'answer': serialized_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
