# import necessary libraries
import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import google.generativeai as genai

# load enviromental variables from file
load_dotenv()

# setup API keys and URLs
QDRANT_API_KEY = "LQQse4CqbEOZqT6BB31dMdKZePJHm8gjum_VNYlnxRWl4oLWYz3wLA"
QDRANT_URL = "https://c08e3b77-6fa2-4aee-953c-a560d95308a0.us-east4-0.gcp.cloud.qdrant.io"
GOOGLE_API_KEY = "AIzaSyCZ1bv7E5UWSiKc2MDdvJf2WqEDYAOb3X0"

print(QDRANT_URL)

# set up embeddings model
MODEL = 'paraphrase-MiniLM-L6-v2'

# Set up LLM model
GEMINI_MODEL = 'gemini-1.5-pro'

COLLECTION_NAME = 'document_collection' 

# intitialize SentenceTransformer model for generating embeddings
embeddings = SentenceTransformer(MODEL)

def embedding_function(text):
    """convert text to embeddings"""
    # check if input is a string or a list of strings
    if isinstance(text, str):
        text = [text]    #convert single string to a list of strings
    # Encode the texts into embeddings and return the first embedding as a list
    return embeddings.encode(text)[0].tolist()

# Initialize Qdrant client for vector database operations
qdrant_client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

# Steps to initialize Qdrant collection:
# 1. Check if it exists
# 2. If not, create a new collection with specified parameters
def initialize_qdrant_collection(collection_name = COLLECTION_NAME, vector_size = 384):
    """Initialize collection in Qdrant if it does not exist yet"""
    # Check if exists
    if not qdrant_client.collection_exists(collection_name):
        # Create a new collection with specified name and vector size
        qdrant_client.create_collection(
            collection_name = collection_name,
            vectors_config = {"content": VectorParams(size = vector_size, distance = Distance.COSINE)}
        )

# Steps to upload documents to qdrant:
# 1. Prepare metadata for each chunk
# 2. Create vector representation of content
# 3. Prepare payload with metadata
# 4. Create PointStruct for Qdrant
# 5. Upload in batches
def upload_documents_to_qdrant(documents, collection_name = COLLECTION_NAME, batch_size = 100):
    """Upload documents in batches to Qdrant vector database"""
    chunked_metadata = []

    for item in documents:
        # prepare meta data for chunk
        id = str(uuid4())    # Generate unique id for each chunk
        content = item.page_content    # Get content of chunk
        source = item.metadata["source"]    #Get source of chunk
        page = item.metadata["page"]    # Get page number of chunk

        # create vector representaion of chunk
        content_vector = embedding_function(content)    # Convert content to embedding
        vector_dict = {"content": content_vector}    # Prepare vector dictionary

        # prepare payLoad with metadata
        payload = {
            "page_content": content,    # Include content of chunk
            "metadata": {
                "id": id,    # Include unique ID
                "page_content": content,    # Include content again for clarity
                "source": source,    # Include source of document
                "page": page,    # Include page number
            }
        }

        # Create PointStruct for Qdrant
        metadata = PointStruct(id = id, vector = vector_dict, payload = payload) # Prepare PointStruct
        chunked_metadata.append(metadata)    # Add to batch

        # Upload in batches
        if len(chunked_metadata) >= batch_size:
            qdrant_client.upsert(collection_name = collection_name, wait = True, points = chunked_metadata)    # Upload the batch
            chunked_metadata = []    # Reset batch after uploading
    
    # Upload any remaining points
    if chunked_metadata:
        qdrant_client.upsert(collection_name = collection_name, wait = True, points = chunked_metadata)    # Upload any remaining documents

# Steps to load documents from a folder:
# 1. Iterate through files in specified folder
# 2. Load PDFs using pypdfLoader
# 3. Split loaded documents into chunks
# 4 return list of documnent chunks
def load_documents_from_folder(data_folder = "data"):
    """Load and split PDFs from a folder"""
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)    # Construct full path to the PDF
            loader = PyPDFLoader(pdf_path)    # Load the pdf
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 50)    # Prepare the text splitter
            documents.extend(loader.load_and_split(text_splitter))    # Load, split, and add document chunks to list
    return documents

class GeminiLLM:
    """Class to initialize and use the Google Gemini LLM"""
    def __init__(self):
        genai.configure(api_key = GOOGLE_API_KEY)    # Configure Gemini with API key
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def generate_response(self, prompt):
        """Generate a response using the Gemini model"""
        response = self.model.generate_content(prompt)    # Generate content based on the prompt
        return response.text    # Return the generated response text
    
# Steps to setup vectorstore and retriever
# 1. Initialize Qdrant vectorstore with client, collection name, and embedding function
# 2. Create a retriever form the vectorstore with defined search parameters
def setup_vectorstore_and_retriever(collection_name = COLLECTION_NAME):
    """Setup Qdrant vector store and retriever"""
    vectorstore = Qdrant(
        client = qdrant_client,
        collection_name = collection_name,
        embeddings = embedding_function,
        vector_name = "content" 
    )    #Initianlize the vectorstore

    retriever = vectorstore.as_retriever(search_kwargs = {"k": 2})    # Create a retriever with search parameters
    return retriever

# Template for chatbot responses
response_template = """
You are an assistant that answers questions about the Ivy Tech School of Information technology.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say you don't know

Contect: {context}

Question: {question}

Answer:
"""

# Steps to get a response:
# 1. Retrieve relevant documents using the retriever
# 2. Prepare context from retrieved documents
# 3. Create a prompt with context and question
# 4. Generate a respone using Gemini LLM
def get_response(question, retriever, LLm_choice = 'gemini'):
    """Retrieve relevant documents and generate a response using Gemini LLM."""
    docs = retriever.get_relevant_documents(question)    # Retrieve relevant documents

    # Prepare context from documents
    context = "\n".join([doc.page_content for doc in docs])    # Join content of retrieved documents

    # Create the prompt with context and question
    prompt = response_template.format(context = context, question = question)    # Format the prompt template

    # Get response from Gemini
    gemini_llm = GeminiLLM()    # Initialize the Gemini LLM
    response = gemini_llm.generate_response(prompt)    # Generate a response based on the prompt

    return context, response

# Main execution block
if __name__ == "__main__":
    # Initialize Qdrant collection
    initialize_qdrant_collection()

    # Load and upload documents to Qdrant
    documents = load_documents_from_folder()
    upload_documents_to_qdrant(documents)

    # Setup retriever
    retriever = setup_vectorstore_and_retriever()

    # Print total points of the vectorDB
    total_points = qdrant_client.get_collection(COLLECTION_NAME).points_count
    print(f"Total points in the vector database: {total_points}")

    # Chatbot loop
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower == 'quit':
            break

        context, response = get_response(question, retriever)
        print(f"\nQuestion: {question}")
        print("'''''''''''''")
        print(f"Context: {context}")
        print("'''''''''''''")
        print(f"Answer: {response}")
        print("'''''''''''''\n")