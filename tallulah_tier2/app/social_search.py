import os
import streamlit as st
from googleapiclient.discovery import build  
import requests  
import pandas as pd
from streamlit_player import st_player
from apify_client import ApifyClient
import pickle
import time
import subprocess
from pytube import Channel, Playlist, YouTube
import whisper
from audiorecorder import audiorecorder
import openai





st.set_page_config(layout="wide") 

google_api_key = os.environ["google_api_key"]
youtube = build('youtube', 'v3', developerKey=google_api_key)
video_link_prefix = "https://www.youtube.com/watch?v="


client = ApifyClient(os.environ["apify_api_key"])

with st.spinner("Loading models"):
    transcription_model = whisper.load_model("base")


if 'youtube_response_videos' not in st.session_state:
    st.session_state['youtube_response_videos'] = []

if 'tiktok_response_videos' not in st.session_state:
    st.session_state['tiktok_response_videos'] = []

if "tiktok_mapping" not in st.session_state:
    st.session_state["tiktok_mapping"] = {}

if "query" not in st.session_state:
    st.session_state["query"] = "kidneycancer" 

if "transcription" not in st.session_state:
    st.session_state["transcription"] = {}

if "transcription_video" not in st.session_state:
    st.session_state["transcription_video"] = {}


if 'reddit_responses' not in st.session_state:
    st.session_state['reddit_responses'] = {}


if 'gpt_responses' not in st.session_state:
    st.session_state['gpt_responses'] = {}


if os.path.isfile("data/default_tiktok_response_videos.pkl"):
    with open('data/default_tiktok_response_videos.pkl', 'rb') as f:
        st.session_state['tiktok_response_videos'] = pickle.load(f)

if os.path.isfile("data/default_tiktok_mapping.pkl"):
    with open("data/default_tiktok_mapping.pkl", 'rb') as f:
        st.session_state['tiktok_mapping'] = pickle.load(f)

if os.path.isfile("data/default_youtube_response_videos.pkl"):
     with open("data/default_youtube_response_videos.pkl", 'rb') as f :
        st.session_state['youtube_response_videos'] = pickle.load(f)


if os.path.isfile("data/transcription_mapping.pkl"):
     with open("data/transcription_mapping.pkl", 'rb') as f :
        st.session_state['transcription'] = pickle.load(f)

if os.path.isfile("data/transcription_video_mapping.pkl"):
     with open("data/transcription_video_mapping.pkl", 'rb') as f :
        st.session_state['transcription_video'] = pickle.load(f)


if os.path.isfile("data/reddit_responses.pkl"):
     with open("data/reddit_responses.pkl", 'rb') as f :
        st.session_state['reddit_responses'] = pickle.load(f)


tiktok_video_folder = "data/tiktok_video_folder"
os.makedirs(tiktok_video_folder, exist_ok = True)


def download_youtube(video_page_url: str, output_path : str) -> None:
    filename = video_page_url.split("v=")[-1]

    if os.path.isfile(output_path + "/" + filename + ".3gpp"):
        return output_path + "/" + filename + ".3gpp"

    youtube_object = YouTube(video_page_url).streams.first()
    #youtube_object = youtube_object.streams.get_highest_resolution()
    if youtube_object is None:
        raise RuntimeError("An error has occurred while getting highest resolution")
    try:
        # output_path = os.path.dirname(path_file_target)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #filename = os.path.basename(path_file_target)
        youtube_object.download(output_path=output_path, filename=filename + ".3gpp")
    except Exception as e:
        raise RuntimeError("An error has occurred while downloading")

    if not os.path.isfile(output_path + "/" + filename + ".3gpp"):
        raise RuntimeError("video does not exist after download")

    print("Download is completed successfully")
    return output_path + "/" + filename + ".3gpp"


def download_video(url, folder, extension=".mp4"):
    if "youtu" in url:
        return download_youtube(url, folder)
    if url in  st.session_state["tiktok_mapping"]:
        return  st.session_state["tiktok_mapping"][url]
    # Send a GET request  
    response = requests.get(url, stream=True)  
    # Check if the request is successful  
    if response.status_code == 200:  
        # Get the file name from the url  
        file_name = url.split("/")[-1]  
        if os.path.isfile(f"{folder}/{file_name}{extension}"):
            return folder + "/" + file_name + extension
        # Open the file in write and binary mode  
        with open(f"{folder}/{file_name}{extension}", 'wb') as file:  
            # Write the content into the file  
            for chunk in response.iter_content(chunk_size=1024):  
                if chunk:  
                    file.write(chunk)  
        return folder + "/" + file_name + extension



def main():
    st.sidebar.markdown(
        f'<img src="https://arrayinsights.com/wp-content/themes/array/images/logo.png" alt="Logo" style="width: 50px;">',
        unsafe_allow_html=True
    )

  #  st.sidebar.image('https://secureailabs.com/wp-content/themes/sail/images/logo.png')
    st.sidebar.title("Tallulah Tier 2")
    page = st.sidebar.selectbox("", ["Social Search", "Story via Video", "Story via Audio", "Translate to Structured"])
    
    if page == "Social Search":
        social_search()
    elif page == "Story via Video":
        story_via_video()
    elif page == "Story via Audio":
        story_via_audio()
    elif page == "Translate to Structured":
        gpt_structured()



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
    # print("Info" + prompt)
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


def gpt_structured():
    input_paragraph = st.text_area("Enter the text to gather insights")
    prompt = st.text_area( "Enter a prompt", '''Reply only in json
                        Leave  keys if not known
                        Use the follwing keys: `Patient Name`, `Patient Age`, `Primary Diagnoses`, `Events`, `Institutions`, `Summary'
                        Under the `Events` key list all events in the patients treatment add `Date` if known"
                        In every event under the key `Emotion` also list a assement of the emotional state of the patient during that event"
                        Under the key `Institutions` list all institutions and organisations the patient interacted with"
                        Under the key "Summary" write a summary of the key Events using the  Emotion for the tone in the sentence. Make the Summary a compelling and cohesive life story'''
    )
    extract = st.button("Extract Insights")

    if extract:
        input_value = input_paragraph + "\n" + prompt
        response = gpt_answer(input_value)
        st.json(response)





def story_via_audio():
    st.header("Record an Audio")
    audio = audiorecorder("Click to record", "Recording...")
    name_audio = "data/audio.mp3"
    #st.text_input("File name for rec")

    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio.export().read())
        
        # To save audio to a file:
        # wav_file = open(name_audio, "wb")
        # wav_file.write(audio.tobytes())
        audio.export(name_audio, format="mp3")
        response = transcription_model.transcribe(name_audio)
        st.text_area("Transcription", response["text"], height=500)
        


def extract_audio(path_file_input: str, path_file_output: str, extension=".mp3"):
    if os.path.isfile(path_file_output + extension):
        return path_file_output + extension

    if not os.path.isfile(path_file_input):
        raise RuntimeError("Input file does not exist")
    output_path = os.path.dirname(path_file_output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        subprocess.run(
            ["ffmpeg", "-i", path_file_input, "-vn", "-acodec", "libmp3lame", "-y", path_file_output + extension],
            check=True,
        )
        print("Audio extracted and saved as:", path_file_output)
    except subprocess.CalledProcessError as e:
        print("Error occurred while extracting audio:", e)
    return path_file_output + extension

def extract_transcript(url):  
    video_file = download_video(url, "data/transcription_folder/video")
    audio_name = video_file.split("/")[-1].replace(".3gpp", "").replace(".mp4","")
    audio_file = extract_audio(video_file, "data/transcription_folder/audio/" + audio_name)
    result = transcription_model.transcribe(audio_file)
    return result["text"], video_file
   

def story_via_video():
    url = st.text_input("Enter URL link")
    transcribe = st.button("Transcribe")
    if transcribe:
        if url in st.session_state.transcription:
            response = st.session_state.transcription[url]
            video_file = st.session_state["transcription_video"][url]
        else:
            with st.spinner('Transcribing...'):
                response, video_file = extract_transcript(url)
                st.session_state.transcription[url] = response
                st.session_state["transcription_video"][url] = video_file
  
    if url in st.session_state.transcription:
        # st.info( st.session_state.transcription_video[url] )
        if "youtu" in url:
            st_player(url)
        else:
            st.video(st.session_state["transcription_video"][url] )
        st.text_area("Transcription", st.session_state.transcription[url], height=700)
    
    if len(st.session_state["transcription"]) > 0:
        with open('data/transcription_mapping.pkl', 'wb') as outp:
            pickle.dump(st.session_state["transcription"], outp)
    
    if len(st.session_state["transcription_video"]) > 0:
        with open('data/transcription_video_mapping.pkl', 'wb') as outp:
            pickle.dump(st.session_state["transcription_video"], outp)




   
def social_search():
    query = st.text_input("Enter your search terms", value=st.session_state["query"])
    search = st.button("Social Search")


    if search:
        ## Youtube
        
        if query != st.session_state["query"] or len(st.session_state['youtube_response_videos']) == 0:
            request = youtube.search().list(
                    q=query,  
                    part="snippet",  
                    type="video",  
                    maxResults=3,
                    key=google_api_key
            )
            response_video = request.execute()
            st.session_state['youtube_response_videos'] = response_video
            
            

        ## Tiktok
        run_input = {
        "hashtags": query.split(","),
        "resultsPerPage": 3,
        "shouldDownloadVideos": True,
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshowImages": False,
        "videoKvStoreIdOrName": "mytiktokvideos",
        "disableEnrichAuthorStats": False,
        "disableCheerioBoost": False,
        } 
        if query != st.session_state["query"] or len(st.session_state['tiktok_response_videos']) == 0:
            data = []
            with st.spinner('Searching...'):
                run = client.actor("clockworks/free-tiktok-scraper").call(run_input=run_input)
                for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                    data.append(item)
            # with open('tiktok_sample_dataset.json') as json_file:
            #     data = json.load(json_file)
                st.session_state['tiktok_response_videos'] = data
        
        #st.session_state["query"] = query


        # ## Reddit

        run_input_reddit = run_input = {
                                "searches": query.split(","),
                                "skipComments": False,
                                "searchPosts": True,
                                "searchComments": True,
                                "searchCommunities": False,
                                "searchUsers": False,
                                "maxItems": 3,
                                "maxPostCount": 3,
                                "maxComments":3,
                                "maxCommunitiesCount": 2,
                                "maxUserCount": 2,
                                "scrollTimeout": 40,
                                "proxy": {"useApifyProxy": True},
                                "debugMode": False,
        }
        if query != st.session_state["query"] or len(st.session_state['reddit_responses']) == 0:
            data = []
            with st.spinner('Searching...'):
                run = client.actor("trudax/reddit-scraper-lite").call(run_input=run_input_reddit)
                for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                    data.append(item)
            # with open('tiktok_sample_dataset.json') as json_file:
            #     data = json.load(json_file)
                st.session_state['reddit_responses'] = data
        
        # st.info(data)
        
        
        st.session_state["query"] = query

        youtube_con = st.expander("Youtube", expanded=True)
        tiktok = st.expander("Tiktok", expanded=True)
        reddit = st.expander("Reddit", expanded=True)
        videos_per_row = 3
        if len(st.session_state['youtube_response_videos']) > 0:
            with open('data/youtube_response_videos.pkl', 'wb') as outp:
                pickle.dump(st.session_state['youtube_response_videos'], outp)
            with youtube_con:
                response_video = st.session_state['youtube_response_videos']
                df = pd.json_normalize(response_video["items"])
                df["video_link"] = video_link_prefix + df["id.videoId"]
                cols = st.columns(videos_per_row)
                i = 0
                for j in range(videos_per_row):
                    with cols[j]: 
                        index = i* videos_per_row + j
                        if index < len(df):
                            st.markdown(f"**Title:** {df['snippet.title'][index]}")  
                            st.markdown(f"**Description:** {df['snippet.description'][index]}")  
                            st.markdown(f"**Published At:** {df['snippet.publishedAt'][index]}")  
                            st.markdown(f"**Channel:** {df['snippet.channelTitle'][index]}")  
                            st.markdown(f"**URL:** {df['video_link'][index]}") 
                            st_player(df['video_link'][index])



        with tiktok:
            if len(st.session_state['tiktok_response_videos']) > 0:
                with open('data/tiktok_response_videos.pkl', 'wb') as outp:
                    pickle.dump(st.session_state['tiktok_response_videos'], outp)
                response_video = st.session_state['tiktok_response_videos']
                view_tt = []
                for item in response_video:
                    view_tt.append((item["authorMeta"]["name"],
                                item["authorMeta"]["following"],
                                item["authorMeta"]["fans"], 
                                item["authorMeta"]["video"],
                                item["text"],
                                item["webVideoUrl"],
                                item["videoMeta"]["downloadAddr"],
                                [temp["name"] for temp in item["hashtags"]]))
                df = pd.DataFrame(view_tt, columns=["name", "following", "fans", "number_videos", "text", "video", "download_video", "hashtags"])
                df["download_location"] = "temp"
                if len(st.session_state["tiktok_mapping"]) > 0:
                    df["download_location"] = list(st.session_state["tiktok_mapping"].values())

                cols_tt = st.columns(videos_per_row)
                i = 0
                for j in range(videos_per_row):
                    with cols_tt[j]: 
                        index = i* videos_per_row + j
                        if index < len(df):
                            st.markdown(f"**Name:** {df['name'][index]}")  
                            st.markdown(f"**Description:** {df['text'][index][0:50] + '...'}")  
                            st.markdown(f"**Hashtags** {df['hashtags'][index][0:10]}")   
                            video_file = df["download_location"][index]
                            st.markdown(f"**Download URL:** {df['download_video'][index]}") 
                            with st.spinner('Video...'):
                                download_link = df['download_video'][index]
                                # st.info(tiktok_video_folder)
                                video_file = download_video(download_link, tiktok_video_folder)
                                # st.info(video_file)
                                df["download_location"][index] = video_file
                                st.session_state["tiktok_mapping"][download_link] = video_file
                                
                            if video_file != "temp":
                                st.video(video_file)
                            #st.info(df["video"][index])
                if len(st.session_state["tiktok_mapping"]) > 0:
                    with open('data/tiktok_mapping.pkl', 'wb') as outp:
                        pickle.dump(st.session_state["tiktok_mapping"], outp)
        with reddit:
            if len(st.session_state['reddit_responses']) > 0:
                with open('data/reddit_responses.pkl', 'wb') as outp:
                    pickle.dump(st.session_state['reddit_responses'], outp)
            reddit = st.session_state['reddit_responses']
            view_tt = []
            for item in reddit:
                view_tt.append([item['username'], 
                               item['body'], item["url"]
                    ])
            
            df = pd.DataFrame(view_tt, columns=["name", "body", "url"])
            cols_tt = st.columns(videos_per_row)
            i = 0
            for j in range(videos_per_row):
                with cols_tt[j]:
                    index = i* videos_per_row + j
                    if index < len(df):
                        st.markdown(f"**Name:** {df['name'][index]}") 
                        st.markdown(f"**Content:** {df['body'][index]}")
                        st.markdown(f"**URL:** {df['url'][index]}")
                



if __name__ == "__main__":
    main()