api_key = "AIzaSyBkxwWO74DmFDgvXwAGPUvwymTRhoYWrtM"

import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Set Hugging Face token
os.environ["API_KEY"] = api_key
genai.configure(api_key=os.environ["API_KEY"])

# Initialize Hugging Face embeddings
embedding_function = HuggingFaceEmbeddings()
client = chromadb.EphemeralClient()

# Load PDF document
book_path = "/Users/sahilbarbade/RAG/document/ugrulebook.pdf"
loader = PyPDFLoader(book_path)
document = loader.load()

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True
)

chunks = splitter.split_documents(document)

# Create Chroma database from chunks
db = Chroma.from_documents(
    chunks, embedding_function, client=client, collection_name="iitb_collection"
)

# Set up query and search for similar documents
query = "What is a registration?"
docs = db.similarity_search(query, k=2)
message = "\n\n -- \n\n".join([l.page_content for l in docs])

# Create prompt template
PROMPT_TEMPLATE = """Answer the questions based on the following context: \n\n {context} \n --- \n\n Answer the questions based on above context \n\n {query} """
prompts = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompts.format(context=message, query=query)

# Initialize Hugging Face question-answering pipeline
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(prompt)
print(response.text)