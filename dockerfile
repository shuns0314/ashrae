FROM python:3.7.5-buster

WORKDIR /code
COPY ./ashrae /code
RUN pip install -r requirements.txt
CMD ["/bin/bash"]