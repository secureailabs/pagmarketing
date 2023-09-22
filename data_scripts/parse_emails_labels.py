import json
import streamlit as st


json_file = st.file_uploader("Choose Email JSON file")
label = st.text_input("Class Label")
dataset_id = st.text_input("Dataset ID")
output_file_name = st.text_input("Output File Name")

if json_file:
    data = json.load(json_file)

parse = st.button("Parse Data and Save")
push_list = []

if parse:
    for email_info in data:
        sample = {}
        sample['dataset_id'] = dataset_id
        sample['text'] = email_info["subject"] + "\n" + email_info["body"]
        sample['text_label'] = label
        push_list.append(sample)

    temp_value = {"list_instance": push_list}
    #parsed_data = json.dumps(temp_value)


    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump(temp_value, f, ensure_ascii=False, indent=4)
    st.info("Done")
