from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Ensure the correct import path
from langchain_community.vectorstores import Chroma  # Ensure the correct import path
import bs4

# Load PDF document
loader = PyPDFLoader("test.pdf")

# Check if the loader returns text directly or requires additional processing
text = loader.load()

# Split text into documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(text)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings()  # Ensure correct initialization if any parameters are required
db = Chroma.from_documents(docs[:20], embeddings)

# Perform similarity search
query = "What are we looking for?"
print(db.similarity_search(query)[0].page_content)
