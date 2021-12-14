FROM python:3.8.12-slim

ENV PORT=5000
EXPOSE 5000

# start to install backend-end stuff
RUN mkdir -p /app
WORKDIR /app

# Install Python requirements.
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pip install pipenv
RUN pipenv install --deploy --system

# Install Python requirements.
COPY ["fer2013_server.py", "./"]
COPY ["models/model1extra.tflite", "./models/"]

# Start server
#ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:$PORT", "fer2013_server:app"]
CMD gunicorn fer2013_server:app --bind 0.0.0.0:$PORT
