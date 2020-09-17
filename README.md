# User-Empowered Anime Recommender

For my [Metis](https://www.thisismetis.com/data-science-bootcamps) final project, I built an anime recommender that combines content-based and collaborative filtering. 

The recommender is deployed on a Flask app and allows users to choose how adventurous they want their recommendations to be. Setting the recommendation type to "more adventurous" on the app will place a greater weight on collaborative filtering recommendations and vice versa. By allowing the user to select their recommendation type, they can get recommendations that better fit their preferences. 

*Images/data were sourced from the internet for educational purposes.*

## Table of Contents

* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Setup](#setup)
* [Metis](#metis)

## Screenshots

![Flask home page](https://user-images.githubusercontent.com/62628676/93408976-d78cc880-f863-11ea-9978-c921b2a56945.png)
![Flask recommender page](https://user-images.githubusercontent.com/62628676/93409135-33efe800-f864-11ea-9c10-0396cda3428d.png)

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
    * If you want to scrape using the cloud, use Dockerfile in containers/container1 directory.
3. Run app.py in flask directory to start Flask app locally on your machine. 

## Metis 

[Metis](https://www.thisismetis.com/data-science-bootcamps) is a 12-week accredited data science bootcamp where students build a 5-project portfolio. 
