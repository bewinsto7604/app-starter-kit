import streamlit as st
import getpass
import os
import time
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI

openaikey = st.secrets["OPENAI_API_KEY"]
loader = TextLoader("padb-domain.txt")
rec_loader = TextLoader('rules.txt')
expert_loader = TextLoader('Expert-info.txt')
index1_loader = TextLoader('Index-training-1.txt')
index2_loader = TextLoader('Index-training-2.txt')
index3_loader = TextLoader('Index-training-3.txt')
index4_loader = TextLoader('Index-training-4.txt')
explain_loader = TextLoader('explain.txt')
whatif_loader = TextLoader('whatif.txt')

doc = loader.load()
rec_doc = rec_loader.load()
expert_doc = expert_loader.load()
index1_doc = index1_loader.load()
index2_doc = index2_loader.load()
index3_doc = index3_loader.load()
index4_doc = index4_loader.load()
explain_doc = explain_loader.load()
whatif_doc = whatif_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
rec_docs = text_splitter.split_documents(rec_doc)
expert_docs = text_splitter.split_documents(expert_doc)
index1_docs = text_splitter.split_documents(index1_doc)
index2_docs = text_splitter.split_documents(index2_doc)
index3_docs = text_splitter.split_documents(index3_doc)
index4_docs = text_splitter.split_documents(index4_doc)
explain_docs = text_splitter.split_documents(explain_doc)
whatif_docs = text_splitter.split_documents(whatif_doc)

embeddings = OpenAIEmbeddings(openai_api_key=openaikey)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
docsearch = FAISS.from_documents(docs, embeddings)
faiss_index_name = 'faiss_index_incremental'
docsearch.save_local(faiss_index_name)
rec_docsearch = FAISS.from_documents(rec_docs, embeddings)
expert_docsearch = FAISS.from_documents(expert_docs, embeddings)
index1_docsearch = FAISS.from_documents(index1_docs, embeddings)
index2_docsearch = FAISS.from_documents(index2_docs, embeddings)
index3_docsearch = FAISS.from_documents(index3_docs, embeddings)
index4_docsearch = FAISS.from_documents(index4_docs, embeddings)
explain_docsearch = FAISS.from_documents(explain_docs, embeddings)
whatif_docsearch = FAISS.from_documents(whatif_docs, embeddings)

llm=OpenAI(temperature=0, openai_api_key=openaikey)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
rec_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=rec_docsearch.as_retriever())
expert_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=expert_docsearch.as_retriever())
index1_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index1_docsearch.as_retriever())
index2_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index2_docsearch.as_retriever())
index3_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index3_docsearch.as_retriever())
index4_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index4_docsearch.as_retriever())
explain_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=explain_docsearch.as_retriever())
whatif_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=whatif_docsearch.as_retriever())

query = "What is QTXACHG?"
qa.run(query)
# Custom image for the app icon and the assistant's avatar
company_logo = 'https://www.app.nl/wp-content/uploads/2019/01/Blendle.png'
# Configure Streamlit page
st.set_page_config(
    page_title="Your Notion Chatbot",
    page_icon=company_logo
)
# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human! I am Blendle's smart AI. How can I help you today?"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        response = ""
        if "POORLY_PERFORMING_SQL" in query:
            print("enter")
            result = agent_executor.run('"' + query + '"')
            response = result
            print(response)
        else:    
            result = whatif_qa.run(query)
            response = result
        # result = chain({"question": query})
        # response = result['answer']
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})            
