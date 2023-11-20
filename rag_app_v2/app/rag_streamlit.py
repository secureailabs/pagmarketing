import os
import streamlit as st
import openai
import weaviate
import pandas as pd
#from google.colab import drive
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from annotated_text import annotated_text




st.set_page_config(layout="wide")

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

azure_openai_endpoint=os.environ["azure_openai_endpoint"]
azure_openai=os.environ["azure_openai"]
azure_openai_engine=os.environ["azure_openai_engine"]

data_folder = os.environ["data_folder"]

@st.cache_resource
def get_vector_handle():
    qa_chain = None
    data = None
    try:
        client = weaviate.Client(
            url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
        )
  
        llm = AzureChatOpenAI(
            api_key=azure_openai,
            api_version="2023-03-15-preview",
            azure_endpoint=azure_openai_endpoint,
            model=azure_openai_engine,
            temperature=0.7,
            max_tokens=800,
            )

        temp = []
        for name in [data_folder + "/kidney_cancer_stories_v2.txt", data_folder + "/cd_stories_v2.txt", data_folder + "/hcl_stories_v2.txt", data_folder + "/hpv_stories_v2.txt"]:
            df = pd.read_csv(name, index_col=0)
            temp.append(df)
        data = pd.concat(temp)
        data['Name_clean'] = [name.replace("The patient's name is ", "").replace("The patient name in the story is","").replace("Patient name: ","").replace("The patient's name in the story is","").replace("The patient's name in this story is ","").replace("Patient Name: ","").replace("The patient name is ","").strip() for name in data['Name']]
        
        temp_1 = []
        for name in ["flu", "hpv"]:
            df = pd.read_csv(data_folder + '/scrape_' + name + '.csv', index_col=0)
            temp_1.append(df)

        data_reddit = pd.concat(temp_1)

        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs
        )

        all_docs = []

        vector_db = Weaviate.from_documents(
            all_docs, client=client, by_text=False, index_name="Patient_stories", embedding=embeddings
        )

        vector_db_reddit = Weaviate.from_documents(
            all_docs, client=client, by_text=False, index_name="Reddit_stories", embedding=embeddings
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(search_kwargs={"k": 4}), return_source_documents=True, verbose=True
            )
        
        qa_chain_reddit = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_db_reddit.as_retriever(search_kwargs={"k": 4}), return_source_documents=True, verbose=True
            )
    except Exception as e:
        st.info(e)
    
    return qa_chain, data, qa_chain_reddit, data_reddit


qa_chain, data, qa_chain_reddit, data_reddit = get_vector_handle()


question = st.text_input("Enter your question about Patient Storeies: HPV, Kidney Cancer, Hairy Cell  Leukaemia, or Celiac disease or Reddit: HPV or Flu")
option = st.selectbox(
    'Choose database',
    ('Patient Stories', 'Reddit'))

button_rag = st.button("ASK")
if button_rag:  
    if option == 'Patient Stories':
        response = qa_chain("Context: " + question + ". Only use information provided in the context")
        st.info(response['result'])
        rows = []
        size = len(response['source_documents'])
        for source in response['source_documents']:
            #st.info(source)
            if "##" in dict(source)['page_content']:
                name_value, content = dict(source)['page_content'].split("##")
                df = data[(data.Name_clean == name_value) & (data['Story'].str.contains(content)==True)].copy()
                df['Excerpt'] = content
                rows.append(df)

        row_df = pd.concat(rows)
        row_df.drop_duplicates(inplace=True)
        # st.dataframe(row_df[["Name_clean", "Story", "Excerpt"]])

        cols = st.columns(size)

        for i in range(len(row_df)):
            with cols[i]:
                    st.markdown("**" + row_df.iloc[i]["Name_clean"] + "**")
                    # st.info(row_df.iloc[i])
                    #st.info(row_df.iloc[i]['Excerpt'])
                    start_ind = row_df.iloc[i]['Story'].index(row_df.iloc[i]['Excerpt'])
                    end_ind = start_ind + len(row_df.iloc[i]['Excerpt'])
                    # st.info(start_ind)
                    # st.info(end_ind)
                    #st.info(row_df.iloc[i]['Story'][start_ind:end_ind])
                    if start_ind > 0 :
                        st.markdown(row_df.iloc[i]['Story'][0:start_ind])
                    st.info((row_df.iloc[i]['Story'][start_ind:end_ind]))
                    st.markdown(row_df.iloc[i]['Story'][end_ind:len(row_df.iloc[i]['Story'])])
    
    if option == 'Reddit':
        response = qa_chain_reddit("Context: " + question + ". Only use information provided in the context")
        st.info(response['result'])
        rows = []
        size = len(response['source_documents'])
        for source in response['source_documents']:
            #st.info(source)
            try:
                if "##" in dict(source)['page_content']:
                    name_value, subreddit, content = dict(source)['page_content'].split("##")
                    content_temp = content.replace("(","").replace(")","")
                    df = data_reddit[(data_reddit.author == name_value) & (data_reddit['content'].str.contains(content_temp)==True)].copy()
                    df['Excerpt'] = content
                    rows.append(df)
            except Exception as e:
                print(e)
                continue

        row_df = pd.concat(rows)
        row_df.drop_duplicates(inplace=True)

        cols = st.columns(size)

        for i in range(len(row_df)):
            with cols[i]:
                    st.markdown("**" + row_df.iloc[i]["author"] + "**")
                    st.markdown("**[blue]" + row_df.iloc[i]["subreddit"] + "**")
                    # st.info(row_df.iloc[i])
                    #st.info(row_df.iloc[i]['Excerpt'])
                    start_ind = row_df.iloc[i]['content'].index(row_df.iloc[i]['Excerpt'])
                    end_ind = start_ind + len(row_df.iloc[i]['Excerpt'])
                    # st.info(start_ind)
                    # st.info(end_ind)
                    #st.info(row_df.iloc[i]['Story'][start_ind:end_ind])
                    if start_ind > 0 :
                        st.markdown(row_df.iloc[i]['content'][0:start_ind])
                    st.info((row_df.iloc[i]['content'][start_ind:end_ind]))
                    st.markdown(row_df.iloc[i]['content'][end_ind:len(row_df.iloc[i]['content'])])
