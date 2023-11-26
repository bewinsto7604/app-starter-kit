import streamlit as st
import getpass
import os
import time
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
            result = db.similarity_search(query)
            response = result[0].page_content
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
