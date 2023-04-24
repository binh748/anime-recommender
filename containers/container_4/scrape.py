"""This module contains web scraping helper functions to scrape user anime lists on myanimelist.net."""

from bs4 import BeautifulSoup
import time
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
    """Returns dictionary of data for user's animelist."""
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
