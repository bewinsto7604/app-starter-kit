import streamlit as st
import getpass
import os
import time
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
# from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

username = "dbmasteruser" 
password = st.secrets["DB_PASSWORD"]
host = "ls-33fa4ea7c905e7c94ad71a9651449adfc0d5b2d3.c9pxztxaqz52.us-east-1.rds.amazonaws.com" 
port = "3306"
mydatabase = "BASELINE_STMT_STATISTICS"
mysql_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(mysql_uri)
agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0, model="gpt-4-0125-preview"),
    toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0, model="gpt-4-0125-preview")),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

history = StreamlitChatMessageHistory(key="chat_messages")

history.add_user_message("hi!")
history.add_ai_message("whats up?")

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
# if len(msgs.messages) == 0:
#   msgs.add_ai_message("How can I help you?")

template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

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

embeddings = OpenAIEmbeddings(openai_api_key=openaikey, model="text-embedding-3-large")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
docsearch = FAISS.from_documents(docs, embeddings)
rec_docsearch = FAISS.from_documents(rec_docs, embeddings)
expert_docsearch = FAISS.from_documents(expert_docs, embeddings)
index1_docsearch = FAISS.from_documents(index1_docs, embeddings)
index2_docsearch = FAISS.from_documents(index2_docs, embeddings)
index3_docsearch = FAISS.from_documents(index3_docs, embeddings)
index4_docsearch = FAISS.from_documents(index4_docs, embeddings)
explain_docsearch = FAISS.from_documents(explain_docs, embeddings)
whatif_docsearch = FAISS.from_documents(whatif_docs, embeddings)
rec_docsearch.save_local("faiss_padb_index")

# llm=OpenAI(temperature=0, openai_api_key=openaikey, model="gpt-4")
llm = ChatOpenAI(model_name='gpt-4')
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
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human! I am BMC Apptune statistics utility assistant for DB2. How can I help you today?"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
llm_chain = LLMChain(llm=OpenAI(model_name="gpt-4-0125-preview"), prompt=prompt, memory=memory)
# Chat logic
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        response = ""
        if "Which SQLSTMT" in prompt:
            print("enter")
            result = agent_executor.run('"' + prompt + ' in POORLY_PERFORMING_SQL"')
            response = result
            print(response)
        elif "Which SQL" in prompt:
            print("enter")
            result = agent_executor.run('"' + prompt + ' in POORLY_PERFORMING_SQL"')
            response = result
            print(response) 
        elif "What is" in prompt:
            print("enter")
            result = agent_executor.run('"' + prompt + ' in POORLY_PERFORMING_SQL"')
            response = result
            print(response)             
        else:    
            result = expert_qa.run(prompt)
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
