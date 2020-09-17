"""This is the main web scraping script to scrape MAL user anime lists."""

import time
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
# Scrape is my module that initiates the Chrome driver and contains helper functions
import scrape

# May need to change the file path depending on where the pickle is located
with open('user_ids_to_rescrape.pkl', 'rb') as read_file:
    user_ids = pickle.load(read_file)

# The main part of the web scraping script
for i in tqdm(range(63, 215)):
    # Need to change i back to i*100
    animelist_rescraped_100_chunk = Parallel(n_jobs=4, verbose=5) \
        (map(delayed(scrape.get_animelist_data), user_ids[i*100:i*100+100]))
    scrape.driver.quit()
    with open(f'pickles/animelist_rescraped_100_{i}.pkl', 'wb') as to_write:
        pickle.dump(animelist_rescraped_100_chunk, to_write)
    # Pause for 3 minutes before continuing the for loop
    time.sleep(180)
