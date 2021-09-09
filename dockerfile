FROM python:3.9.2

WORKDIR /usr/src/ML

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app