import streamlit as st
import openai
import weaviate
import pandas as pd
#from google.colab import drive
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain.agents.types import AgentType
from langchain.agents import AgentExecutor, Tool,initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from annotated_text import annotated_text




st.set_page_config(layout="wide")

WEAVIATE_URL = "https://ragtestarray-4gihzxpr.weaviate.network"
WEAVIATE_API_KEY = "7E0Vf7POMdgUkpQfEHj5hPMpfUtPxNCNIisB"

azure_openai_endpoint="https://pagmarketingopenai.openai.azure.com/"
azure_openai="0c5adeb8121b4602b3e0735ba4a06ef9"
azure_openai_engine="pagmarketinggpt"

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
        # for i in client.schema.get()['classes']:
        #     print(i['class'])
        #     client.schema.delete_class(i['class'])
        temp = []
        for name in ["data/kidney_cancer_stories_v2.txt", "data/cd_stories_v2.txt", "data/hcl_stories_v2.txt", "data/hpv_stories_v2.txt"]:
            df = pd.read_csv(name, index_col=0)
            temp.append(df)
        data = pd.concat(temp)
        data['Name_clean'] = name_clean = [name.replace("The patient's name is ", "").replace("The patient name in the story is","").replace("Patient name: ","").replace("The patient's name in the story is","").replace("The patient's name in this story is ","").replace("Patient Name: ","").replace("The patient name is ","").strip() for name in data['Name']]
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        #model_kwargs = {"device": "cuda"} mps
        #model_kwargs = {"device": "mps"}
        model_kwargs = {}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs
        )
        text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
        all_docs = []
        # vectors_docs = []
        # count = 0
        # for index, d in data.iterrows():
        #     try:
        #         try:
        #             base_docs = text_splitter.split_text(d['Story'])
        #             list_docs = [d['Name_clean'] + '##' + base_docs[i] for i in range(0, len(base_docs))]
        #             create_docs = text_splitter.create_documents(list_docs)
        #         except Exception:
        #             list_docs = []
        #             create_docs = []
        #             continue

        #         all_docs.extend(create_docs)

        #         count = count + 1
        #     except Exception as  e:
        #         print(e)
        #         continue
        vector_db = Weaviate.from_documents(
            all_docs, client=client, by_text=False, index_name="Patient_stories", embedding=embeddings
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(search_kwargs={"k": 4}), return_source_documents=True, verbose=True
            )
    except Exception as e:
        print(e)
    
    return qa_chain, data


qa_chain, data = get_vector_handle()


question = st.text_input("Enter your question about HPV, Kidney Cancer, Hairy Cell  Leukaemia, or Celiac disease")
button_rag = st.button("ASK")
if button_rag:  
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



    
