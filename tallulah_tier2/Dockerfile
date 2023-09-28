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
COPY app/ /app/

#RUN apt-get install -y ffmpeg

# RUN wget https://evermeet.cx/ffmpeg/get/zip && unzip -o zip



#RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements.txt
RUN cd arin_pytube && python setup.py install
RUN cd ..

# RUN sh setup.sh

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/social_search.py", "--server.port=8501", "--server.address=0.0.0.0"]
