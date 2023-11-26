import streamlit as st
import getpass
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

loader = TextLoader("padb-domain.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
db = FAISS.from_documents(docs, embeddings)

query = "What is QTXACHG?"
docs = db.similarity_search(query)
print(docs[0].page_content)
