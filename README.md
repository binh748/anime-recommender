# User-Empowered Anime Recommender

For my [Metis](https://www.thisismetis.com/data-science-bootcamps) final project, I built an anime recommender that combines content-based and collaborative filtering. 

The recommender is deployed on a Flask app and allows users to choose how adventurous they want their recommendations to be. Setting the recommendation type to "more adventurous" on the app will place a greater weight on collaborative filtering recommendations and vice versa. By allowing the user to select their recommendation type, they can get recommendations that better fit their preferences. 

To read more, see my [blog post](https://binhhoang.io/blog/anime-recommender/). 

*Images/data were sourced from the internet for educational purposes.*

## Table of Contents

* [Screenshots](#screenshots)
* [Features](#features) 
* [Technologies](#technologies)
* [Setup](#setup)
* [Metis](#metis)

## Screenshots

![Flask home page](https://user-images.githubusercontent.com/62628676/97792286-b8978b00-1bb2-11eb-8a9d-7df79a578d28.png)
![Flask recommender page](https://user-images.githubusercontent.com/62628676/93409135-33efe800-f864-11ea-9c10-0396cda3428d.png)

## Features

* Get anime recommendations for a MyAnimeList user ID **(only works on anime and user IDs scraped from MyAnimeList.net)** 

## Technologies

* Python 3.8
* HTML5
* CSS3
* Flask 1.1.2
* BeautifulSoup 4.9.1
* Selenium 3.141.0
* Docker 19.03.12
* Google Cloud Compute Engine
* MongoDB 4.4.0
* Scikit-learn 0.23.1
* Pandas 1.0.5
* Numpy 1.18.5
* Seaborn 0.10.1

## Setup

1. Clone this repo.
2. Run anime_recommender.py (may take a while due to web scraping, hence better to use a Docker container to deploy web scraper across multiple cloud instances).
    * If you want to scrape using the cloud, use Dockerfile in containers/container_1 directory.
3. Run app.py in flask directory to view Flask app.  

## Metis 

[Metis](https://www.thisismetis.com/data-science-bootcamps) is a 12-week accredited data science bootcamp where students build a 5-project portfolio. 
