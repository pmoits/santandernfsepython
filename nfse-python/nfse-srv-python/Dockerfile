#FROM python:3.7
FROM ubuntu:18.04

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update ##[edited]
RUN apt-get install -y libgl1-mesa-dev

RUN apt-get --fix-missing update && apt-get --fix-broken install && apt-get install -y poppler-utils && apt-get install
RUN apt-get install -y libleptonica-dev && ldconfig && apt-get install -y python3.6
RUN apt-get install -y python3-pip && apt install -y libsm6 libxext6
RUN apt-get install -y tesseract-ocr

COPY model_labels.dat /app
COPY result_model_letter.h5 /app
COPY model_labels_upper.dat /app
COPY result_model_letter_upper.h5 /app
COPY uwsgi.ini /app
COPY result_model_ctc.h5 /app


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app

#EXPOSE 5000

CMD [ "uwsgi", "--ini", "uwsgi.ini"]
#CMD ['python3',"app.py"]