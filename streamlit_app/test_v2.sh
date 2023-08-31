docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.8.0
sudo sysctl -w vm.max_map_count=262144
#docker run -d --name es-node01 --net elastic -p 9200:9200 -p 9300:9300 -t docker.elastic.co/elasticsearch/elasticsearch:8.8.0
docker run -d -eELASTIC_PASSWORD="=hExbx*huLo1-vtaISC=" --name es-node01 --net elastic -p 9200:9200 -p 9300:9300 -t docker.elastic.co/elasticsearch/elasticsearch:8.8.0
docker cp es-node01:/usr/share/elasticsearch/config/certs/http_ca.crt app/
elastic_ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' es-node01)

elastic_url="https://${elastic_ip}:9200"
patient_index="patient_stories_v2"

docker build -t pag_marketing_streamlit .

docker run -d -e azure_openai="azure_openai_key" \
-e ELASTIC_PASSWORD="ELASTIC_PASSWORD" \
-e aws_access_key_id="aws_secret_access_key" \
-e aws_secret_access_key="aws_secret_access_key" \
-e bing_key="bing_key" \
-e bing_web_ep="https://api.bing.microsoft.com/v7.0/search" \
-e bing_visual_ep="https://api.bing.microsoft.com/bing/v7.0/images/visualsearch" \
-e azure_openai_endpoint="https://adamsexperiment.openai.azure.com/" \
-e elastic_url="${elastic_url}" \
-e patient_index="${patient_index}" \
-e azure_openai_engine="model_deployment_name"
-e opencage_key="opencage_key"
--name pag-node01 --net elastic -p 8501:8501 pag_marketing_streamlit
# docker cp http_ca.crt pag-node01:/app/
curl http://0.0.0.0:8501


# https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue

elastic_ip=$(sudo docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' es-node01)

sudo docker run -d -e azure_openai="f358cac4f944406e90abe9f1b69ce651" -e ELASTIC_PASSWORD="=hExbx*huLo1-vtaISC=" -e aws_access_key_id="AKIAVOD5X2XTVVGHUF34" -e aws_secret_access_key="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" -e bing_key="eb8ebc70665844bb87bea3552f62b57d" -e bing_web_ep="https://api.bing.microsoft.com/v7.0/search" -e bing_visual_ep="https://api.bing.microsoft.com/bing/v7.0/images/visualsearch" -e azure_openai_endpoint="https://adamsexperiment.openai.azure.com/" -e elastic_url="${elastic_url}" -e patient_index="${patient_index}" --name pag-node01 --net elastic -p 8501:8501 pag_marketing_streamlit