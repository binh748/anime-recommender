"""This module contains data cleaning functions."""
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm


def fix_mismatching_animelist_len(complete_animelist):
    """Returns user's animelists with 0s inputted for animelist_scores where
    the scraper wasn't able to pick up the scores, hence causing a mismatch in
    lengths between animelist_titles and animelist_scores.

    Args:
        complete_animelist: List of dicts of user animelists scraped from MyAnimeList.net.
    """
    # Find user_ids for animelists where the len of animelist_titles != len of animelist_scores
    user_ids_lens_not_matching = [
        animelist['user_id'] for animelist in complete_animelist
        if animelist['animelist_titles'] and
        len(animelist['animelist_titles']) != len(animelist['animelist_scores'])
    ]

    # Fill in scores with 0s to match lengths of titles
    for animelist in complete_animelist:
        if animelist['user_id'] in user_ids_lens_not_matching:
            animelist['animelist_scores'] = list(
                np.zeros(len(animelist['animelist_titles']), dtype=int))

    return complete_animelist

def create_user_score_dicts(complete_animelist, top_1000_anime_titles):
    """Returns list of dicts where there is a key-value pair for every anime title in
    top_1000_anime_titles and the user's corresponding score or 0 if user did not score.

    This sets up the user-ratings matrix for collaborative filtering.

    Args:
        complete_animelist: List of dicts of user animelists scraped from MyAnimeList.net.
        top_1000_anime_titles: List of top 1000 anime titles on MyAnimeList.net.
    """
    # Create augmented complete_animelist: add a key-value pair for each anime
    # title (key) with the user's corresponding score (value)
    augmented_complete_animelist = deepcopy(complete_animelist)
    for animelist in tqdm(augmented_complete_animelist):
        # Only apply on animelists without Nones
        if animelist['animelist_titles']:
            for anime_title, anime_score in \
            zip(animelist['animelist_titles'], animelist['animelist_scores']):
                animelist[anime_title] = anime_score

    # Create list of dicts where each dict has the user_id,
    # animelist_url, and a key-value pair for each anime in top_1000_anime_titles
    user_score_dicts_top_1000_anime = []
    for animelist in tqdm(augmented_complete_animelist):
        user_score_dict = {}
        user_score_dict['user_id'] = animelist['user_id']
        user_score_dict['animelist_url'] = animelist['animelist_url']
        for anime_title in top_1000_anime_titles:
            # If the user has that anime in their animelist, assign the user's score (including
            # '-', which is not a score, but indicates anime is on user's animelist)
            if anime_title in animelist:
                user_score_dict[anime_title] = animelist[anime_title]
            # Otherwise, assign 0, which means the user did not have the anime in the animelist
            else:
                user_score_dict[anime_title] = 0
        user_score_dicts_top_1000_anime.append(user_score_dict)

    return user_score_dicts_top_1000_anime

def create_user_anime_history_df(user_score_dicts_top_1000_anime, top_1000_anime_titles):
    """Returns user_anime_history_df for content-based filtering.

    Args:
        user_score_dicts_top_1000_anime: List of dicts where there is a key-value
        pair for every anime title in top_1000_anime_titles and the user's
        corresponding score or 0 if user did not score.
        top_1000_anime_titles: List of top 1000 anime titles on MyAnimeList.net.
    """
    user_anime_history_df = pd.DataFrame(user_score_dicts_top_1000_anime)

    # Change the df entries to binary values of 0s and 1s where 1 indicates anime
    # is on user's animelist and 0 indicates the opposite
    # Use pandas assignment operators to leverage pandas's vectorization
    for col in tqdm(user_anime_history_df):
        # Skip the first two columns
        if col in top_1000_anime_titles:
            user_anime_history_df.loc[user_anime_history_df[col] != 0,
                                      col] = 1

    return user_anime_history_df

def clean_user_anime_history_df(user_anime_history_df, top_1000_anime_titles):
    """Returns cleaned user_anime_history_df for content-based filtering.

    Args:
        user_anime_history_df: DataFrame of user's anime viewing history where
        the entries 1 = watched and 0 = not watched.
        top_1000_anime_titles: List of top 1000 anime titles on MyAnimeList.net.
    """
    # Convert all columns (excluding user_id and animelist_url columns) to numeric dtype
    for col in tqdm(user_anime_history_df):
        if col in top_1000_anime_titles:
            user_anime_history_df[col] = user_anime_history_df[col].astype('int64')

    return user_anime_history_df

def create_user_score_df(user_score_dicts_top_1000_anime):
    """Returns user_score_df as matrix of user-scores for content-based filtering.

    Args:
        user_score_dicts_top_1000_anime: List of dicts where there is a key-value
        pair for every anime title in top_1000_anime_titles and the user's
        corresponding score or 0 if user did not score.
    """
    user_score_df = pd.DataFrame(user_score_dicts_top_1000_anime)
    return user_score_df

def clean_user_score_df(user_score_df, top_1000_anime_titles):
    """Returns cleaned user_score_df for NMF in collaborative filtering.

    Args:
        user_score_df: User-rating matrix.
        top_1000_anime_titles: List of top 1000 anime titles on MyAnimeList.net.
    """
    # Convert all '-' to 0s in user_user_df
    for col in tqdm(user_score_df):
        user_score_df[col] = np.where(user_score_df[col] == '-', 0, user_score_df[col])

    # Convert all score columns (excluding user_id and animelist_url columns) to numeric dtype
    # .astype appears to be faster than pd.to_numeric, which makes sense
    for col in tqdm(user_score_df):
        if col in top_1000_anime_titles:
            user_score_df[col] = user_score_df[col].astype('int64')

    return user_score_df

def clean_top_anime_data_1000_df(top_anime_data_1000_df):
    """Returns a cleaned top_anime_data_1000_df for content-based filtering.

    Args:
        top_anime_data_1000_df: DataFrame of data scraped on 1000 top anime.
    """
    # Remove commas from numbers
    top_anime_data_1000_df['members'] = top_anime_data_1000_df['members'].str.replace(',', '')
    top_anime_data_1000_df['favorites'] = top_anime_data_1000_df['favorites'].str.replace(',', '')

    # Change num_episodes with value Unknown to 0 so that I can convert num_episodes column to an int
    top_anime_data_1000_df.loc[top_anime_data_1000_df['num_episodes'] == 'Unknown', 'num_episodes'] = '0'
    top_anime_data_1000_df['num_episodes'] = top_anime_data_1000_df['num_episodes'].astype('int64')

    # Change appropriate columns into numeric types
    top_anime_data_1000_df['score'] = top_anime_data_1000_df['score'].astype('float64')
    top_anime_data_1000_df['scored_by_num_users'] = top_anime_data_1000_df['scored_by_num_users'].astype('int64')
    top_anime_data_1000_df['rank_score'] = top_anime_data_1000_df['rank_score'].str.replace('#', '').astype('int64')
    top_anime_data_1000_df['rank_popularity'] = top_anime_data_1000_df['rank_popularity'].str.replace('#', '').astype('int64')
    top_anime_data_1000_df['members'] = top_anime_data_1000_df['members'].astype('int64')
    top_anime_data_1000_df['favorites'] = top_anime_data_1000_df['favorites'].astype('int64')

    # Split my aired_dates column into aired_from and aired_to where NaN is inputted where there is no aired_to date
    # since some anime are just movies or single OVA episodes
    top_anime_data_1000_df['aired_from'], top_anime_data_1000_df['aired_to'] = \
        top_anime_data_1000_df['aired_dates'].str.split(' to ', 1).str

    # Set missing aired_to dates to aired_from dates since missing aired_to dates
    # represent movies and other media type where there is only one aired date
    top_anime_data_1000_df['aired_to'] = top_anime_data_1000_df['aired_to'].fillna(top_anime_data_1000_df['aired_from'])

    # Because of a bug in my scraping script, I didn't properly scrape the media_type=Music. Correcting that here.
    top_anime_data_1000_df.loc[top_anime_data_1000_df['media_type'].str.contains('Episodes:'), 'media_type'] = 'Music'

    # Fill missing 'premiered' values with aired_from dates
    top_anime_data_1000_df['premiered'] = top_anime_data_1000_df['premiered'].fillna(top_anime_data_1000_df['aired_from'])

    # Convert aired_from to datetime64 dtype
    top_anime_data_1000_df['aired_from'] = pd.to_datetime(top_anime_data_1000_df['aired_from'])

    # Using date 2020-09-07 as today date to calculate age of anime in years
    top_anime_data_1000_df['age_in_years'] = (pd.to_datetime('today') - top_anime_data_1000_df['aired_from']).dt.days / 360

    return top_anime_data_1000_df
