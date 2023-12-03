from langchain.memory import StreamlitChatMessageHistory
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
# from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
import random
company_logo = 'https://www.app.nl/wp-content/uploads/2019/01/Blendle.png'
username = "dbmasteruser" 
password = st.secrets["DB_PASSWORD"]
host = "ls-33fa4ea7c905e7c94ad71a9651449adfc0d5b2d3.c9pxztxaqz52.us-east-1.rds.amazonaws.com" 
port = "3306"
mydatabase = "BASELINE_STMT_STATISTICS"
mysql_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(mysql_uri)
agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)),
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
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

# Add the memory to an LLMChain as usual
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

import streamlit as st
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    response = llm_chain.run(prompt)
    # response = agent_executor.run('"' + prompt + '"')
    st.chat_message("ai").write(response)
