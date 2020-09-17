# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7

WORKDIR /usr/src/app

# Install manually all the missing libraries
RUN apt-get update
RUN apt-get install -y wget gconf-service libasound2 libatk1.0-0 libcairo2 libcups2 libfontconfig1 libgdk-pixbuf2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libxss1 fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils

# Install Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN dpkg -i google-chrome-stable_current_amd64.deb; apt-get -fy install

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Make a pickles directory to store my pickles
RUN mkdir /usr/src/app/pickles

# Copy local code to the container image
COPY main.py ./
COPY scrape.py ./
COPY user_ids_to_rescrape.pkl ./

# Run my script when I run Dockerfile
CMD [ "python", "./main.py" ]
