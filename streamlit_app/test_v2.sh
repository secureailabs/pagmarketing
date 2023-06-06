docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.8.0
#docker run -d --name es-node01 --net elastic -p 9200:9200 -p 9300:9300 -t docker.elastic.co/elasticsearch/elasticsearch:8.8.0
docker run -d -eELASTIC_PASSWORD="=hExbx*huLo1-vtaISC=" --name es-node01 --net elastic -p 9200:9200 -p 9300:9300 -t docker.elastic.co/elasticsearch/elasticsearch:8.8.0
docker cp es-node01:/usr/share/elasticsearch/config/certs/http_ca.crt app/

elastic_ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' es-node01)

elastic_url="https://${elastic_ip}:9200"

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
--name pag-node01 --net elastic -p 8501:8501 pag_marketing_streamlit
# docker cp http_ca.crt pag-node01:/app/
curl http://0.0.0.0:8501
