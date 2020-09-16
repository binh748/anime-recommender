"""This is the main Python file to create the anime recommender system from scratch."""

import time
import pickle
from tqdm import tqdm
from src import data_cleaning as dc, recommender as rec, scrape
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import NMF

# These will be all the users included in my recommender system. In this project,
# I got data on 120,000 users.
base_url = 'https://myanimelist.net/users.php'
mal_user_ids_urls = scrape.get_mal_user_ids_urls(base_url, num_users=240)
user_ids = scrape.get_mal_user_ids(mal_user_ids_urls)

# Scrape user animelists
for i in tqdm(range(1200)):
    animelist_data_100_chunk = Parallel(n_jobs=4, verbose=5) \
        (map(delayed(scrape.get_animelist_data), user_ids[i*100:i*100+100]))
    scrape.driver.quit()
    with open(f'../pickles/animelist_data_100_{i}.pkl', 'wb') as to_write:
        pickle.dump(animelist_data_100_chunk, to_write)
    # Pause for 3 minutes to not ping the web server too much
    time.sleep(180)

# Concatenating all the animelist_data_chunks into a complete_animelist
complete_animelist = []
for i in tqdm(range(1200)):
    with open(f'../pickles/animelist_data_100_{i}.pkl', 'rb') as read_file:
        animelist_data_chunk = pickle.load(read_file)
    complete_animelist += animelist_data_chunk

# Scrape data on 1,000 of top anime
mal_ids_top_1000_anime = scrape.get_top_anime_mal_ids(num_top_anime=1000)
top_anime_data_1000 = [
    scrape.get_animelist_data(user_id) for user_id in tqdm(mal_ids_top_1000_anime)
]
top_anime_data_1000_df = pd.DataFrame(top_anime_data_1000)
top_1000_anime_titles = top_anime_data_1000_df['title_main'].to_list()

complete_animelist = dc.fix_mismatching_animelist_len(complete_animelist)

# Create dfs for content-based and collaborative filtering
user_score_dicts_top_1000_anime = dc.create_user_score_dicts(complete_animelist, top_1000_anime_titles)
user_anime_history_df = dc.create_user_anime_history_df(user_score_dicts_top_1000_anime, top_1000_anime_titles)
user_anime_history_df = dc.clean_user_anime_history_df(user_anime_history_df, top_1000_anime_titles)
user_score_df = dc.create_user_score_df(user_score_dicts_top_1000_anime)
user_score_df = dc.clean_user_score_df(user_score_df, top_1000_anime_titles)
top_anime_df = dc.clean_top_anime_data_1000_df(top_anime_data_1000_df)

# Use NMF to create user/a/nime embeddings for collaborative-filtering
nmf = NMF(n_components=6, max_iter=500, random_state=4444)
user_embedding = nmf.fit_transform(user_score_df.drop(columns=['user_id', 'animelist_url']))

user_embedding_df = pd.DataFrame(user_embedding.round(2))
anime_embedding_df = pd.DataFrame(nmf.components_.round(2),
                                  columns=top_1000_anime_titles)
