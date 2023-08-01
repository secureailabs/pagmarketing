### PAG MArketing Demo

##APIs used
* BING Web Search
* BING Visual Search
* Amazon Rekognition
* AZure Cognitive Services (Open AI Chat GPT)

#Secrets
All secrets to be set up under .streamlit/secrets.toml as key-value pairs (key=value and one per line)

## Requirements
* Set up elastic search using docker - https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
* sudo sysctl -w vm.max_map_count=262144 (elastic search)
* https://stackoverflow.com/questions/66351223/repl-it-smtplib-oserror-errno-99-cannot-assign-requested-address (on Azure)

## TODO
Dockerize this application and have it on same network as elastic search


