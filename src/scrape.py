"""This module contains functions to scrape myanimelist.net."""

import random
import re
import time
from bs4 import BeautifulSoup
from jikanpy import Jikan
import requests
from fake_useragent import UserAgent
import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException


# Define Chrome browser options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless") # Hides the browser window
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("window-size=1024,768")
chrome_options.add_argument("--no-sandbox")
# Suggested by StackOverflow. This line of code is absolutely crucial; otherwise,
# my Chrome driver crashes in a docker container. Another workaround is to allocate
# more RAM to my docker container at runtime.
# chrome_options.add_argument('--disable-dev-shm-usage')

# Initialize the driver (aka a new broswer)
driver = webdriver.Chrome(options=chrome_options)


def create_soup(url):
    """Returns BeautifulSoup object for given URL."""
    ua = UserAgent()
    user_agent = {'User-agent': ua.random}
    response_text = requests.get(url, headers=user_agent).text
    soup = BeautifulSoup(response_text, 'html5lib')
    return soup


def create_soup_selenium(url, driver=driver):
    """Returns BeautifulSoup object for given URL by using Selenium to load
    the webpage in chromedriver and extract the HTML."""
    driver.get(url)
    # time.sleep gives time for the webpage to fully load
    # and then I can extract the HTML
    time.sleep(3)
    # Wait for the 'table' element to appear on
    # the page before extracting the HTML
    try:
        # Wait until the page is loaded before returning the soup
        WebDriverWait(driver, 5) \
            .until(expected_conditions.visibility_of_element_located((By.TAG_NAME, 'table')))
        soup = BeautifulSoup(driver.page_source, 'html5lib')
        return soup
    # If TimeoutException, means the animelist is restricted and so will
    # return the soup without confirming there is a 'table' element
    except TimeoutException:
        # print("""It's possible that the anime list is restricted or
        # the page is not loading for some reason.
        # Will return the soup and move on.""")
        soup = BeautifulSoup(driver.page_source, 'html5lib')
        return soup
        # try:
        #     # If 'badresult' class exists, means the anime list is private
        #     # and cannot be accessed. Will return soup as is.
        #     if driver.find_element_by_class_name('badresult'):        #
        # except NoSuchElementException:
        #     driver.refresh()


def get_animelist_data(user_id):
    """Returns dictionary of key data from user's animelist."""
    BASE_URL = 'https://myanimelist.net/animelist/'
    url = BASE_URL + user_id
    soup = create_soup_selenium(url, driver)
    animelist_data = {
        'user_id': user_id,
        'animelist_url': url,
        'animelist_titles': get_animelist_titles(soup),
        'animelist_scores': get_animelist_scores(soup)
    }
    return animelist_data


def get_animelist_titles(soup):
    """Returns list of all anime titles in user's animelist."""
    if soup.find_all('tbody', class_='list-item'):
        animelist_titles = []
        for element in soup.find_all('tbody', class_='list-item'):
            # Making sure that the element exists before appending
            if element.find(class_='data title clearfix'):
                if element.find(class_='data title clearfix').find(class_='link sort'):
                    animelist_title = element.find(class_='data title clearfix') \
                        .find(class_='link sort').text
                    animelist_titles.append(animelist_title)
        return animelist_titles


def get_animelist_scores(soup):
    """Returns list of all anime ratings in user's animelist."""
    if soup.find_all('tbody', class_='list-item'):
        animelist_scores = []
        for element in soup.find_all('tbody', class_='list-item'):
            # Making sure that the element exists before appending
            if element.find(class_='data score'):
                if element.find(class_='data score').text:
                    animelist_score = element.find(class_='data score').text.strip()
                    animelist_scores.append(animelist_score)
        return animelist_scores


def get_anime_data(mal_id):
    """Returns dictionary of key data for anime."""
    BASE_URL = 'https://myanimelist.net/anime/'
    url = BASE_URL + str(mal_id)
    soup = create_soup(url)
    time.sleep(0.5+2*random.random())
    anime_data = {
        'mal_id': mal_id,
        'url': url,
        'image_url': get_image_url(soup),
        'trailer_url': get_trailer_url(soup),
        'title_main': get_title_main(soup),
        'title_english': get_title_english(soup),
        'media_type': get_media_type(soup),
        'source_material': get_source_material(soup),
        'num_episodes': get_num_episodes(soup),
        'airing_status': get_airing_status(soup),
        'aired_dates': get_aired_dates(soup),
        'premiered': get_premiered(soup),
        'duration': get_duration(soup),
        'content_rating': get_content_rating(soup),
        'genres': get_genres(soup),
        'score': get_score(soup),
        'scored_by_num_users': get_scored_by_num_users(soup),
        'rank_score': get_rank_score(soup),
        'rank_popularity': get_rank_popularity(soup),
        'members': get_members(soup),
        'favorites': get_favorites(soup),
        'studios': get_studios(soup),
        'producers': get_producers(soup),
        'licensors': get_licensors(soup)
    }
    return anime_data


def get_image_url(soup):
    if soup.find('img', itemprop='image'):
        image_url = soup.find('img', itemprop='image').get('data-src')
        return image_url


def get_trailer_url(soup):
    if soup.find(class_='iframe js-fancybox-video video-unit promotion'):
        trailer_url = soup.find(class_= \
            'iframe js-fancybox-video video-unit promotion').get('href')
        return trailer_url


def get_title_main(soup):
    if soup.find(class_='title-name'):
        title_main = soup.find(class_='title-name').text
        return title_main


def get_title_english(soup):
    if soup.find('span', text='English:'):
        raw_text = soup.find('span', text='English:').findParent().text
        # Clean raw text to extract the English title
        title_english = raw_text.strip().replace('English: ', '')
        return title_english


def get_media_type(soup):
    if soup.find('span', text='Type:'):
        media_type = soup.find('span', text='Type:').findNext().text
        return media_type


def get_source_material(soup):
    if soup.find('span', text='Source:'):
        raw_text = soup.find('span', text='Source:').findParent().text
        # Clean raw text
        source_material = raw_text.strip().replace('Source:', '').strip()
        return source_material


def get_num_episodes(soup):
    if soup.find('span', text='Episodes:'):
        raw_text = soup.find('span', text='Episodes:').findParent().text
        # Clean raw text
        num_episodes = raw_text.strip().replace('Episodes:', '').strip()
        return num_episodes


def get_airing_status(soup):
    if soup.find('span', text='Status:'):
        raw_text = soup.find('span', text='Status:').findParent().text
        # Clean raw text
        airing_status = raw_text.strip().replace('Status:', '').strip()
        return airing_status


def get_aired_dates(soup):
    if soup.find('span', text='Aired:'):
        raw_text = soup.find('span', text='Aired:').findParent().text
        # Clean raw text
        aired_dates = raw_text.strip().replace('Aired:', '').strip()
        return aired_dates


def get_premiered(soup):
    if soup.find('span', text='Premiered:'):
        premiered = soup.find('span', text='Premiered:').findNext().text
        return premiered


def get_duration(soup):
    if soup.find('span', text='Duration:'):
        raw_text = soup.find('span', text='Duration:').findParent().text
        # Clean raw text
        duration = raw_text.strip().replace('Duration:', '').strip()
        return duration


def get_content_rating(soup):
    if soup.find('span', text='Rating:'):
        raw_text = soup.find('span', text='Rating:').findParent().text
        # Clean raw text
        content_rating = raw_text.strip().replace('Rating:', '').strip()
        return content_rating


def get_genres(soup):
    if soup.find('span', text='Genres:'):
        genres = [element.text for element in \
            soup.find('span', text='Genres:').findParent().find_all('a')]
        return genres


def get_score(soup):
    if soup.find('span', itemprop='ratingValue'):
        score = soup.find('span', itemprop='ratingValue').text
        return score


def get_scored_by_num_users(soup):
    if soup.find('span', itemprop='ratingCount'):
        scored_by_num_users = soup.find('span', itemprop='ratingCount').text
        return scored_by_num_users


def get_rank_score(soup):
    if soup.find('span', text='Ranked:'):
        rank_score = soup.find('span', text='Ranked:').findParent() \
            .find(text=re.compile('#')).strip()
        return rank_score


def get_rank_popularity(soup):
    if soup.find('span', text='Popularity:'):
        rank_popularity = soup.find('span', text='Popularity:').findParent() \
            .find(text=re.compile('#')).strip()
        return rank_popularity


def get_members(soup):
    if soup.find('span', text='Members:'):
        raw_text = soup.find('span', text='Members:').findParent().text
        # Clean raw text
        members = raw_text.strip().replace('Members:', '').strip()
        return members


def get_favorites(soup):
    if soup.find('span', text='Favorites:'):
        raw_text = soup.find('span', text='Favorites:').findParent().text
        # Clean raw text
        favorites = raw_text.strip().replace('Favorites:', '').strip()
        return favorites


def get_studios(soup):
    if soup.find('span', text='Studios:'):
        studios = [element.text for element in \
            soup.find('span', text='Studios:').findParent().find_all('a')]
        return studios


def get_producers(soup):
    if soup.find('span', text='Producers:'):
        producers = [element.text for element in \
            soup.find('span', text='Producers:').findParent().find_all('a')]
        return producers


def get_licensors(soup):
    if soup.find('span', text='Licensors:'):
        licensors = [element.text for element in \
            soup.find('span', text='Licensors:').findParent().find_all('a')]
        return licensors


def get_mal_user_ids_urls(base_url, num_users=240):
    """Returns list of MyAnimeList URLs containing user IDs.

    Args:
        base_url: URL of MyAnimeList search page for users.
        num_users: Number of users to obtain user IDs for.
    """
    add_urls = []
    # num_users-24 because the first page already includes 24 users
    for i in range(24, num_users-24+1, 24):
        add_urls.append(f'{base_url}&show={i}')
    urls_list = [base_url] + add_urls
    return urls_list


def get_mal_user_ids(urls):
    """Returns list of MyAnimeList user IDs.

    Args:
        urls: List of MyAnimeList URLs containing user IDs.
    """
    user_ids = []
    page_counter = 0
    for url in urls:
        page_counter += 1
        print(page_counter)
        soup = create_soup(url)
        for element in soup.find_all(href=re.compile('/profile/')):
            # "if element.text" removes any cases where the href does not contain the user's ID
            if element.text:
                user_ids.append(element.text)
    return user_ids


def get_top_anime_mal_ids(num_top_anime=1000):
    """Returns list of MyAnimeList anime IDs for anime listed on top anime pages.

    Args:
        num_top_anime: Number of top anime to scrape anime IDs for.
    """
    jikan = Jikan()
    counter = 0
    mal_ids = []
    num_top_anime_pages = num_top_anime // 50 # 50 anime per top anime page
    for i in range(1, num_top_anime_pages+1): # +1 because range does not include stop
        result_page = jikan.top(type='anime', page=i)['top']
        time.sleep(2+2*random.random())
        counter += 1
        print(counter)
        for result in result_page:
            mal_ids.append(result['mal_id'])
    return mal_ids
