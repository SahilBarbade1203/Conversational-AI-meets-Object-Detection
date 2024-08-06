import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Set Hugging Face token
os.environ["HUGGINGFACE_API_TOKEN"] = "--your api_key---"

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
query = "What if a student fails in 5 courses?"
docs = db.similarity_search(query, k=2)
message = "\n\n -- \n\n".join([l.page_content for l in docs])

# Create prompt template
PROMPT_TEMPLATE = """Answer the questions based on the following context: \n\n {context} \n --- \n\n Answer the questions based on above context \n\n {query} """
prompts = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompts.format(context=message, query=query)

# Initialize Hugging Face question-answering pipeline
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small" ,truncation=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Generate answer
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens = 500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
