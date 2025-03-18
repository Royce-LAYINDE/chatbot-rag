

FROM python:3.10

WORKDIR /exam_byRoyce

COPY requirements.txt /exam_byRoyce/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /exam_byRoyce

CMD ["uvicorn", "chatbotDIT:app", "--host", "0.0.0.0", "--port", "8000"]

