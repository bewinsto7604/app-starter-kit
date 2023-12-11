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
openai.api_key = st.secrets["OPENAI_API_KEY"]
f = open('zcv1-171494-dba-mapping.json')
mapping = json.load(f)
mapping_string = json.dumps(mapping)
query1 = "When cond_code which is not equal to 0 it denotes a failed job. Show me jobname, utilname, error_reason_cd, error_reason for all failed jobs for event.utilname BMC AMI Reorg for Db2 in the last 2 days."
query2 = "Show me all failed jobs for BMC AMI Reorg for Db2 in the last 3 days"
def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0)

    return response.choices[0].message["content"]
    

prompt = 'Given the mapping delimited by triple backticks ``` ' + mapping_string + ' ``` translate the text delimited by triple quotes in a valid Elasticsearch DSL query """ ' + query1 + query2 + ' """. Give me only the json code part of the answer. Compress the json output removing spaces.'
response = get_completion(prompt)

print(response)

