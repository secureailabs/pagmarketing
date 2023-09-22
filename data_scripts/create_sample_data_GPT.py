import openai
import os
import pandas as pd
import sys
import json
import streamlit as st
import re

N = 2

def gpt_answer(prompt):

    openai.api_type = "azure"
    openai.api_base = os.environ["azure_openai_endpoint"]
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.environ['azure_openai']

    response = openai.ChatCompletion.create(engine=os.environ['azure_openai_engine'],
                                                messages = [{"role": "user", "content": prompt}],
                                                temperature=0.7,
                                                max_tokens=2000,
                                                top_p=0.95,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=None)
    
    result = response.choices[0]["message"]["content"]
    
    return result


# def process_real_data(df, prompt_base):
#     response_list = ""
#     #df = pd.read_csv(input_file, sep=",") 
#     #print(df.head(1))
#     for index, row in df.iterrows(): 
#         print("Processing row: " + str(index)) 
#         story =  row["Stories"]
#         tags = row["Tags"] 
#         # prompt = "Create " + str(N) +" samples similar this story: " + story + "with tags: " + tags +". Make the output into JSON format"
#         prompt =  prompt_base.format(story=story, tags=tags)
#         st.info(prompt)
#         response = gpt_answer(prompt)
#         response_list =  response_list + response
#     return response_list


# file_name = st.file_uploader("Choose the sample data file")
# output_file = st.text_input("Enter output file name", value="sample_output.json")
# if file_name:
#     df = pd.read_csv(file_name, sep=",")
#     st.dataframe(df)

def split_numbered_list(text):  
    # Split the text using regex, pattern matches any digit followed by a period '.'  
    pattern = r'\d+\.'  
    # Split the string using the pattern, maxsplit = 0 means unlimited splits  
    split_text = re.split(pattern, text, maxsplit=0)  
    # Remove any empty strings from the list  
    split_text = list(filter(None, split_text))  
    return split_text


prompt = st.text_input("Enter Prompt" , value ="")
N = st.number_input("Number of times", min_value=1, step=1)
process = st.button("Save")

if process:
    all_responses = []
    for i in range(0, N):
        print(str(i))
        response = gpt_answer(prompt)
        # st.info(response)
        data = json.loads(response)
        all_responses = all_responses + data
        #st.info(len(data))
        st.info(len(all_responses))
    
    with open('data/data.json', 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)


# if __name__ == "__main__":
#     process_real_data(sys.argv[1], sys.argv[2])