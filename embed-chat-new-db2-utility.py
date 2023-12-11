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
from langchain.chains.LLMChain
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import openai
st.set_page_config(page_title="DB2 Utility assistant", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

f = open('zcv1-171494-dba-mapping.json')
mapping = json.load(f)
mapping_string = json.dumps(mapping)

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

prompt = 'Given the mapping delimited by triple backticks ``` ' + mapping_string + ' ``` translate the text delimited by triple quotes in a valid Elasticsearch DSL query """ ' + query2 + ' """. Give me only the json code part of the answer. Compress the json output removing spaces.'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0)
    return response.choices[0].message["content"]
    
response = get_completion(prompt)

print(response)
