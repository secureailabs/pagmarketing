FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

#RUN mkdir /app/data

COPY app/ /app/
#COPY data/ /app/data

RUN pip3 install -r requirements.txt
RUN cd ..


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/rag_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
