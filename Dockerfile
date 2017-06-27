FROM python:slim

COPY . ./

RUN pip3 install -r requirements.txt
RUN mkdir waves

CMD ["python3", "main.py"]