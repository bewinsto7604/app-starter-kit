import openai
import os
import pandas as pd
import time
import requests
import json
import streamlit as st
import getpass
import os
import time
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

f = open('zcv1-171494-dba-mapping.json')
mapping = json.load(f)
mapping_string = json.dumps(mapping)

# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

# template / prompt
query1 = "When cond_code which is not equal to 0 it denotes a failed job. Show me jobname, utilname, error_reason_cd, error_reason for all failed jobs for event.utilname BMC AMI Reorg for Db2 in the last 2 days."
query2 = "Show me all failed jobs for BMC AMI Reorg for Db2 in the last 3 days"
template = """I want you to act as a customer service assistant for a
BMC Utility for Db2. Respond in a friendly and helpful tone, with very concise answers.
Make sure to ask the user relevant follow up questions.

Given the context, these are the allowed cond_code
When cond_code which is not equal to 0 it denotes a failed job. Always include jobname, utilname, error_reason_cd, error_reason in the output
{context}

{chat_history}
Human: {human_input}
Chatbot:"""

str_prompt = 'Given the mapping delimited by triple backticks ``` ' + mapping_string + ' ``` translate the text delimited by triple quotes in a valid Elasticsearch DSL query """ ' + query2 + ' """. Give me only the json code part of the answer. Compress the json output removing spaces.'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0)
    return response.choices[0].message["content"]

# Initialization
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key="chat_history", input_key="human_input"
    )
company_logo = 'https://www.app.nl/wp-content/uploads/2019/01/Blendle.png'
# Configure Streamlit page
st.set_page_config(
    page_title="Your Notion Chatbot",
    page_icon=company_logo
)
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human! I am smart AI. How can I help you today?"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
# llm_chain = LLMChain(llm=OpenAI(), prompt=str_prompt, memory=memory)
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
        if "Show me" in prompt:
            print("enter")
            response = get_completion(query1)
            print(response)
        elif "Which SQL" in prompt:
            print("enter")
            result = agent_executor.run('"' + prompt + ' in POORLY_PERFORMING_SQL"')
            response = result
            print(response)    
        else:    
            result = whatif_qa.run(prompt)
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
