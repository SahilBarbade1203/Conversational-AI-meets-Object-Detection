# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
# actions/actions.py

api_key = "AIzaSyBkxwWO74DmFDgvXwAGPUvwymTRhoYWrtM"

import os
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Set Hugging Face token
os.environ["API_KEY"] = api_key
genai.configure(api_key=os.environ["API_KEY"])

class ActionQueryRAG(Action):

    def name(self) -> str:
        return "action_query_rag"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> list:
        
        # Load the document and setup the environment (you might want to optimize this)
        book_path = "/Users/sahilbarbade/RAG/document/RASA/books/ugrulebook.pdf"
        loader = PyPDFLoader(book_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100,
            add_start_index = True
        )

        chunks = splitter.split_documents(document)

        embedding_function = HuggingFaceEmbeddings()
        client = chromadb.EphemeralClient()

        db = Chroma.from_documents(
            chunks, embedding_function, client=client, collection_name="iitb_collection"
        )

        # Get the user query
        query = tracker.latest_message.get('text')

        docs = db.similarity_search(query, k = 2)
        message = "\n\n -- \n\n".join([l.page_content for l in docs])

        PROMPT_TEMPLATE = """Answer the questions based on following context: \n\n {context} \n --- \n\n Answer the questions based on above context \n\n {query} """
        prompts = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompts.format(context = message, query = query)

        # Initialize google gemini flash model for question-answering pipeline
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        # Send the response back to the user
        dispatcher.utter_message(text=response.text)

        return []

