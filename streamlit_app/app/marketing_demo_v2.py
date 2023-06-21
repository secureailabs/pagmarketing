import os
import boto3
import json
import streamlit as st
from streamlit_chat import message
from elasticsearch import Elasticsearch
import openai
import PyPDF2
import docx2txt
import pandas as pd
import requests
from PIL import Image
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from sentence_transformers import SentenceTransformer
import email
import imaplib
from email.header import decode_header


## Keep session between pages

# Create Elasticsearch client
#es = Elasticsearch("https://localhost:9200")  # Update with your Elasticsearch server URL
es = Elasticsearch(
    os.environ["elastic_url"],
  #  ssl_assert_fingerprint=os.environ['CERT_FINGERPRINT'],
    ca_certs="http_ca.crt",
    basic_auth=("elastic", os.environ['ELASTIC_PASSWORD'])

)

es.options(request_timeout=3600)
patient_index = os.environ["patient_index"]

@st.cache_resource
def create_index():
    try:
        es.indices.create(index=patient_index,
                            mappings= {
                                "properties": {
                                    "Picture_image_embedding": {
                                        "type": "dense_vector",
                                        "dims": 512,
                                        "index": True,
                                        "similarity": "cosine"
                                        },
                                        "Picture_source": {
                                             "type": "keyword"
                                        }
                                        }
                                        }
                            )
    except Exception as e:
        print(str(e))

# Create directory for images
images_dir = 'images'
if not os.path.exists(images_dir):
   os.makedirs(images_dir)


email_dir = 'email'
if not os.path.exists(email_dir):
   os.makedirs(email_dir)

if not os.path.exists(email_dir + "/images"):
   os.makedirs(email_dir + "/images")

if not os.path.exists(email_dir + "/docs"):
   os.makedirs(email_dir + "/docs")

docs_dir = 'docs'
if not os.path.exists(docs_dir):
   os.makedirs(docs_dir)

# Survey questions
questions = [
    ("Name", "text", {}, ''),
    ("Age", "number", {"min_val": 5, "max_val": 90, "step": 1}, ''),
    ("Life Story", "textarea", {}, ''),
    ("Tags", "text", {}, "(comma delimited words)"),
    ("Picture", "file", {"type": ["png", "jpeg", "jpg"]}, ''),
    ("Additional Documents", "file", {"type": ["doc", "docx", "txt", "pdf"]}, '')
]

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


#load model
@st.cache_resource
def get_model():
    model = SentenceTransformer('clip-ViT-B-32')
    return model

#cached functions
get_model()
create_index()

def process_image(uploaded_file):
    # Create Amazon Rekognition client
    rekognition = boto3.client(
    "rekognition",
    region_name="us-east-1",
    aws_access_key_id=os.environ['aws_access_key_id'],
    aws_secret_access_key=os.environ['aws_secret_access_key']
    )  
    # Send image to Amazon Rekognition for facial analysis
    image = uploaded_file.read()
    response = rekognition.detect_labels(
        Image={"Bytes": image},
        MaxLabels=25
    )
    labels = [label["Name"].replace("\'", '\"') for label in response["Labels"] if label["Confidence"] > 80]
    
    response = rekognition.detect_faces(
        Image={"Bytes": image},
        Attributes=["ALL"]
    )

    image_info = {}
    image_info["general"] = labels
    image_info["age"] = {"Low": response["FaceDetails"][0]["AgeRange"]["Low"], "High": response["FaceDetails"][0]["AgeRange"]["High"]}
    image_info["gender"] = {"Value": response["FaceDetails"][0]["Gender"]["Value"], "Confidence": response["FaceDetails"][0]["Gender"]["Confidence"]}
    image_info["face_occulded"] = {"Value": response["FaceDetails"][0]["FaceOccluded"]["Value"],"Confidence": response["FaceDetails"][0]["FaceOccluded"]["Confidence"]}
    image_info["emotions"] = [d["Type"].replace("'", '"') for d in response["FaceDetails"][0]["Emotions"] if d["Confidence"] > 80]
    return image_info

## We could add OCR and stuff. Amazon API is weird but maybe the Azure one will work
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    # doc = docx.Document(docx_file)
    # text = ""
    # for paragraph in doc.paragraphs:
    #     text += paragraph.text
    text = docx2txt.process(docx_file)
    return text


def extract_text_from_document(uploaded_file):
    extracted_text = ''
    if '.pdf' in uploaded_file.name:
        extracted_text = extract_text_from_pdf(uploaded_file)
    if ('.doc' in uploaded_file.name):
        extracted_text = extract_text_from_docx(uploaded_file)
    return extracted_text

## Main Page Code

logo_image = "https://secureailabs.com/wp-content/themes/sail/images/logo.png"

def main():
    st.sidebar.markdown(
        f'<img src="https://secureailabs.com/wp-content/themes/sail/images/logo.png" alt="Logo" style="width: 50px;">',
        unsafe_allow_html=True
    )

  #  st.sidebar.image('https://secureailabs.com/wp-content/themes/sail/images/logo.png')
    st.sidebar.title("PAG Patient Storybank")
    page = st.sidebar.selectbox("", ["Patient Intake", "Story Assistant", "Search", "Find the Story", "Email Processing"])
    # "Patient Stories"
    if page == "Patient Intake":
        display_survey()
    # elif page == "Patient Stories":
    #     list_surveys()
    elif page == "Search":
        search_surveys()
    elif page == "Story Assistant":
        gpt_chat()
    elif page == "Find the Story":
        bing_search()
    elif page == "Email Processing":
        email_processing()

# Streamlit app
def display_survey():
    # st.session_state['previous_page'] = "Patient Intake Form"
    st.title("Patient Intake Form")

    ##Track when people are using the stories (Make a story page?)
    ##Search History - number of times a word has ben searched

    # Display survey questions
    survey_responses = {}
    for question, question_type, options, display_info in questions:
        if question_type == "text":
            response = st.text_input(question + display_info)
        elif question_type == "number":
            response = st.number_input(question + display_info, min_value=options.get("min_val", 0), max_value=options.get("max_val", 100), step=options.get("step", 1))
        elif question_type == "textarea":
            response = st.text_area(question + display_info)
        elif question_type == "file":
            response = st.file_uploader(question + display_info, type=options.get("type", ["png", "jpeg", "jpg"]), key=question)
        else:
            response = st.text_input(question)

        if response:
            if question == 'Tags':
                keywords = response.split(",")
                survey_responses[question] = [key.strip() for key in keywords]
            elif options.get("type", None) == ["png", "jpeg", "jpg"]:
                image_info = json.loads(json.dumps(process_image(response)))
                location = images_dir + "/" + response.name
                with open(os.path.join(location),"wb") as f: 
                    f.write(response.getbuffer())  
                embedding = get_model().encode(Image.open(location))
                survey_responses[question + "_source"] = location
                survey_responses[question + "_age"] = image_info["age"]
                survey_responses[question + "_gender"] = image_info["gender"]
                survey_responses[question + "_emotions"] = image_info["emotions"]
                survey_responses[question + "_face_occulded"] = image_info["face_occulded"]
                survey_responses[question + "_labels"] = image_info["general"]
                survey_responses["Picture_image_embedding"] = embedding.tolist()
            elif options.get("type", None) == ["doc", "docx", "txt", "pdf"]:
                doc_info = extract_text_from_document(response)
                survey_responses[question + "_source"] = response.name
                survey_responses[question + "_content"] =  doc_info
            else:
                survey_responses[question] = response

                

    # Submit button
    if st.button("Submit"):
        print(survey_responses)
        # Index survey responses in Elasticsearch
        index_survey_responses(survey_responses, index=patient_index)

        # Clear survey responses
        survey_responses.clear()
        st.success("Survey submitted successfully!")


def index_survey_responses(survey_responses, index="default"):
    # Index the survey responses in Elasticsearch
    es.index(index=index, body=survey_responses)


def list_surveys(search_results):
    for hit in search_results:
       if "Tags" in hit["_source"]:
           if type(hit["_source"]["Tags"]) == str:
               hit["_source"]["Tags"] = hit["_source"]["Tags"].split(',')

    # Display surveys in a table
    
    if search_results:
        df = pd.DataFrame([hit["_source"] for hit in search_results])
        if "Picture_age" in df.columns:
            df["Picture_age"] = df["Picture_age"].astype(str)
        if "Picture_gender" in df.columns:
            df["Picture_gender"] = df["Picture_gender"].astype(str)
        if "Picture_face_occulded" in df.columns:
            df["Picture_face_occulded"] = df["Picture_face_occulded"].astype(str)

        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()

        grid_table = AgGrid(df, height=250, gridOptions=gridoptions,
                            update_mode=GridUpdateMode.NO_UPDATE)
        # st.write('## Selected')
        # selected_row = grid_table["selected_rows"]
        # st.dataframe(selected_row)
    else:
        st.info("No surveys found.")


def search_es(query, index):
    # Search in Elasticsearch
    search_results = es.search(index=index, body={"query": {"query_string": {"query": query}}})
    hits = search_results["hits"]["hits"]
    return hits


def search_surveys():
    st.header("Search Database")
    text,image = st.tabs(["Text", "Image"])
    with text:
        # User input for search query
        query = st.text_input("Enter text query")

        options = [patient_index, "chat_assistance"]
        index = st.selectbox("Choose database", options)
    
        search_but = st.button("Search", key="text")

        if search_but:
            try:
                search_results = search_es(query, index)
                list_surveys(search_results)
            except Exception as e:
                st.info(str(e))
    with image:
        dir_path = "images"
        res = []

        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                res.append(dir_path + "/" + path)

        query = st.selectbox("Find Image",  options=res)
        st.image(Image.open(query))
        #value=st.session_state['previous_bing_search'])
        search = st.button("Search", key="image")
        if search:
            try:
                ## Need to make the URL one work
                if ('.png' in query or '.jpg' in query or '.jpeg' in query):
                    image_info = es.search(
                        index=patient_index,
                        query={
                            "term": {
                                "Picture_source": {
                                    "value": query,
                                    "boost": 1.0
                                }
                            }
                        },
                        source=True)
                    if (image_info is not None):
                        found_image = image_info['hits']['hits'][0]["_source"]
                        found_image_embedding = found_image['Picture_image_embedding']
                        es_query = {
                             "field": "Picture_image_embedding",
                             "query_vector": found_image_embedding,
                             "k": 5,
                             "num_candidates": 10
                             }
                       # print(es_query)
                        response = es.search(
                                    index=patient_index,
                                    fields=["Name", "Picture_source"],
                                knn=es_query, source=False)
                        print(response)
                        search_results = response["hits"]["hits"]
                        row_size = 3
                        controls = st.columns(row_size)
                        col = 0
                        for result in search_results:
                            with controls[col]:
                                if result["_score"] > 0.9:
                                     st.header(result["fields"]["Name"][0])
                                     st.image(Image.open(result["fields"]["Picture_source"][0]))
                            col = (col + 1) % row_size
            except Exception as e:
                st.info(str(e))
                    


def gpt_chat():
    st.header("Story Assistant")

    openai.api_type = "azure"
    openai.api_base = os.environ["azure_openai_endpoint"]
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.environ['azure_openai']

    # st.info(os.environ["azure_openai_endpoint"])
    # st.info(os.environ['azure_openai'])
 

    response = openai.ChatCompletion.create(
        engine=os.environ['azure_openai_engine'],
        messages = [{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    prompt = st.text_area("Enter your prompt")
    run_but = st.button("Run")
    # message("Hello! I am your story assistant. How may I help you?", is_user=False)


    #if st.button("Chat"):
    if prompt or run_but:
        print(prompt)
        response = openai.ChatCompletion.create(engine=os.environ['azure_openai_engine'],
                                                messages = [{"role": "user", "content": prompt}],
                                                temperature=0.7,
                                                max_tokens=800,
                                                top_p=0.95,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=None)
        result = response.choices[0]["message"]["content"]
        try:
            index_survey_responses({"prompt": prompt, "response": result}, index="chat_assistance")
        except Exception as e:
            st.info(str(e))
        st.session_state.past.append(prompt)
        st.session_state.generated.append(result)

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


def bing_search():
    # add autocomplete
    # https://github.com/microsoft/bing-search-sdk-for-python/blob/main/samples/rest/BingWebSearchV7.py
    st.header("Find My Story")
    text_endpoint = os.environ['bing_web_ep']
    visual_endpoint = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
    #os.environ['bing_visual_ep']
    subscription_key = os.environ['bing_key']
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    image_search, text_search = st.tabs(["image_search", "text_search"])
    with image_search:
        list_of_images = []
        # list to store files
        dir_path = "images"
        res = []

        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                res.append(dir_path + "/" + path)

        query = st.selectbox("Find Image",  options=res)
        #value=st.session_state['previous_bing_search'])
        search = st.button("Search")
        if search:
            ## Need to make the URL one work
            if ('.png' in query or '.jpg' in query or '.jpeg' in query):
                # image_query = open(images_dir + "/" + query, 'rb')
                # st.image(Image.open(images_dir + "/" + query))
                image_query = open(query, 'rb')
                st.image(Image.open(query))
                file = {'image': ('MY-IMAGE', image_query)} 
                params = {"knowledgeRequest" : {"invokedSkills": ["SimilarImages"],}}
                try:
                    response = requests.post(visual_endpoint, headers=headers, files=file,params=params)
                    print(response.request)
                    response.raise_for_status()
                    result = response.json()
                    #st.session_state['previous_bing_search_results'] = result
                # print(result.keys())
                    tag_actions_of_interest = ["PagesIncluding", "VisualSearch"]
                    list_items = []
                    if result['tags']:
                        first_tag = result['tags'][0]
                        print("Visual search tag count: {}".format(len(result['tags'])))

                        for tag in result['tags']:
                            if  tag['actions']:
                            #    st.write("First tag action count: {}".format(len(tag['actions'])))
                                for action in tag['actions']:
                                    if (action["actionType"] in tag_actions_of_interest) and ("data" in action):
                                        
                            #         st.write("First tag action type: {}".format(action.keys()))
                                        for val in action["data"]['value']:
                                            row_val = {}
                                            print(val)
                                            if val["isFamilyFriendly"]:
                                                row_val["action_type"] = action["actionType"]
                                                row_val["name"] = val["name"]
                                                row_val["thumbnailUrl"] = val["thumbnailUrl"]
                                                row_val["contentUrl"] = val["contentUrl"]
                                                row_val["hostPageUrl"] = val["hostPageUrl"]
                                            list_items.append(row_val)

                        tabs = st.tabs(tag_actions_of_interest)
                        df = pd.DataFrame(list_items)
                    # required_columns = ["contentUrl", "hostPageUrl", "thumbnailUrl"]
                        for i in range(0, len(tag_actions_of_interest)):
                            with tabs[i]:
                                st.dataframe(
                                    df[df["action_type"] == tag_actions_of_interest[i]],
                                    use_container_width=True,
                                    column_config={
                                    "url": st.column_config.LinkColumn("page link"),
                                    "contentUrl": st.column_config.ImageColumn("image link"),
                                    "hostPageUrl": st.column_config.LinkColumn("host page link"),
                                    "thumbnailUrl": st.column_config.ImageColumn("thumbnail link"),
                                    }
                                    )

                        
                    else:
                        print("Couldn't find image tags!")
                #  print(response.json())
                    #st.dataframe(pd.DataFrame(response.json()))
                except Exception as ex:
                    raise ex
    with text_search:
        query = st.text_input("Find")
        search = st.button("Search", key="text_search")
        print("here 1")
        # Construct a request
        if search:
            mkt = 'en-US'
            params = { 'q': query, 'mkt': mkt, 'safeSearch':'strict'}
            try:
                response = requests.get(text_endpoint, headers=headers, params=params)
                response.raise_for_status()
                #st.session_state['previous_bing_search_results'] = response.json()
                # st.write("\nHeaders:\n")
                # st.write(response.headers)

                #st.write("\nJSON Response:\n")
                #st.json(response.json())
                items = {"webPages":["name", "url", "snippet"], 
                        "images": ["name", "contentUrl", "hostPageUrl", "thumbnailUrl"],
                        "news":["name", "url", "description"]
                        }
                tab_list = st.tabs(items)
                for index, key in enumerate(items):
                    if key in response.json():
                        with tab_list[index]:
                            st.header(key)
                            info = response.json()[key]["value"]
                            df = pd.DataFrame(info)
                            required_columns = items[key]
                            st.dataframe(
                                df[required_columns],
                                use_container_width=True,
                                column_config={
                                "url": st.column_config.LinkColumn("page link"),
                                "contentUrl": st.column_config.LinkColumn("image link"),
                                "hostPageUrl": st.column_config.LinkColumn("host page link"),
                                "thumbnailUrl": st.column_config.LinkColumn("thumbnail link"),
                }
                                )
            except Exception as ex:
                raise ex


def gpt_answer(prompt):

    openai.api_type = "azure"
    openai.api_base = os.environ["azure_openai_endpoint"]
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.environ['azure_openai']
    # response = openai.ChatCompletion.create(
    #     engine="SAIL_Demo",
    #     messages = [{"role": "user", "content": "Hello!"}],
    #     temperature=0.7,
    #     max_tokens=800,
    #     top_p=0.95,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     stop=None)
    
    response = openai.ChatCompletion.create(engine=os.environ['azure_openai_engine'],
                                                messages = [{"role": "user", "content": prompt}],
                                                temperature=0.7,
                                                max_tokens=800,
                                                top_p=0.95,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=None)
    result = response.choices[0]["message"]["content"]
    
    return result
     
 
    
def email_processing():
    with st.form("my_form"):
        username = st.text_input("Gmail user")
        password = st.text_input("Password", type="password")
        categories_list = st.text_input("Categories", value="showing support,clinical trial,recently diagnosed,None", placeholder="showing support,clinical trial,recently diagnosed,None")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Extract and Analyze")
        if submitted:
            cat_list = categories_list.split(',')
            # st.write(cat_list)
            categories_str = [str(i + 1)  + ". " + cat_list[i].strip() for i in range(0, len(cat_list))]
            categories_str = ' or '.join(categories_str)
            # st.write(categories_str)
            SERVER = 'imap.gmail.com'
            imap_port = 993

            # connect to the server and go to its inbox
            mail = imaplib.IMAP4_SSL(SERVER, imap_port)
            mail.login(username + "@gmail.com", password)
            # we choose the inbox but you can select others
            mail.select('inbox')

            # we'll search using the ALL criteria to retrieve
            # every message inside the inbox
            # it will return with its status and a list of ids
            # status, data = mail.search(None, 'UNSEEN')
            status, data = mail.search(None, 'ALL')
            mail_ids = []
            for block in data:
                # the split function called without parameter
                # transforms the text or bytes into a list using
                # as separator the white spaces:
                # b'1 2 3'.split() => [b'1', b'2', b'3']
                mail_ids += block.split()
            # now for every id we'll fetch the email
            # to extract its content
            email_info = []
            for i in mail_ids:
                # the fetch function fetch the email given its id
                # and format that you want the message to be
                status, response = mail.fetch(i, '(RFC822)')
                
                raw_email = response[0][1]
                # st.write(raw_email)
                email_message = email.message_from_bytes(raw_email)
                files_list = []
                mail_content = ''
                content_obj = email_message.get_payload()[0]
               
                if content_obj.get_content_type() == "text/plain":
                    mail_content = content_obj.get_payload(decode=True).decode()

                for i in range(1, len(email_message.get_payload())):
                   # st.write(email_message.get_payload()[i].get_content_type())
                    img = email_message.get_payload()[i]
                    content_type = img.get_content_type()
                    # st.write(img.get_filename())
                    # certain imaes
                    dir = None
                    if "image" in content_type:
                        dir = "images"
                    if "pdf" in content_type:
                        dir = "docs"
                    if "document" in content_type:
                        dir = "docs"

                    if dir:
                        type_file = content_type.split("/")[1]
                        file_name = img.get_filename()
                        filePath = os.path.join(email_dir + "/" + dir, file_name)
                        if not os.path.isfile(filePath):
                            fp = open(filePath, 'wb')
                            fp.write(img.get_payload(decode=True))
                            fp.close()
                        
                        files_list.append(filePath)
                
                # # Extract email information (e.g., subject, sender, date, body)
                mail_subject = decode_header(email_message["Subject"])[0][0]
                mail_from = email_message["From"]
                # # date = email_message["Date"]
                #mail_content = ''
                if email_message.is_multipart():
                     for part in email_message.walk():
                         
                          content_type = part.get_content_type()
                       #   st.write(part.get_payload(decode=True))
                          if content_type == "text/plain":
                              mail_content = part.get_payload(decode=True).decode()
                              break
                else:
                #      # Non-multipart content (plain text or other types)
                      mail_content = email_message.get_payload(decode=True).decode()

                
                
                email_info.append({"from_name": mail_from.split("<")[0], "from_email":mail_from.split("<")[1].replace(">",""),"subject": mail_subject, "content": mail_content, "attachments": files_list})
        
            if len(email_info) > 0:
                st.header("Emails")
                df = pd.DataFrame(email_info)
                st.dataframe(df)

                st.header("Categorize")
                st.caption("Email are categorized into " + categories_str)
                           #1. showing support or 2. clinical trial or 3. recently diagnosed or 4. None")
                categorized = []
                try:
                    for i in range(0, len(email_info)):
                        info = email_info[i]
                        # prompt = "Can you tell me if this following content is about 1. showing support or 2. clinical trial or 3. recently diagnosed or 4. None. Here is the content: " + info["content"] + ". Answer from the 4 choices provided and only answer in 2 words"
                        prompt = "Can you tell me if this following content is about " + categories_str + ". Here is the content: " + info["content"] + ". Answer from the 4 choices provided and only answer in 2 words"
                        # st.write(prompt)
                        answer = gpt_answer(prompt)
                       # st.write(answer)
                        info["category"] = answer
                        if answer != "None":
                            categorized.append(info)
                except:
                    st.info("API is overloaded")
                
                df_new = pd.DataFrame(categorized)  
                st.dataframe(df_new[["category", "content"]])

            mail.close()
            mail.logout()

if __name__ == "__main__":
    main()