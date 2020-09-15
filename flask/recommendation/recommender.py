"""This module contains functions for the anime recommender system."""
import pandas as pd
# Considering using default variables for these functions by loading up
# their pickles to avoid typing all the arguments each time

def get_user_scores(user_id, user_score_df):
    """Returns dictionary of all non-zero user scores, including
    user/animelist metadata.

    Args:
        user_id: MyAnimeList user ID.
        user_score_df: User-rating matrix as a DataFrame.
    """
    user_idx = get_user_idx(user_id, user_score_df)
    user_score_dict = {}
    for col in user_score_df[user_score_df['user_id'] == user_id]:
        if user_score_df.at[user_idx, col] != 0:
            user_score_dict[col] = user_score_df.at[user_idx, col]
    return user_score_dict


def get_user_anime_history(user_id, user_anime_history_df):
    """Returns dictionary of all non-zero entries, including
    user/animelist metadata.

    A non-zero entry indicates that that anime is on the user's
    animelist.

    Args:
        user_id: MyAnimeList user ID.
        user_anime_history_df: DataFrame of user's anime viewing history where
        the entries 1 = watched and 0 = not watched.
    """
    user_idx = get_user_idx(user_id, user_anime_history_df)
    user_anime_history_dict = {}
    for col in user_anime_history_df[user_anime_history_df['user_id'] == user_id]:
        if user_anime_history_df.at[user_idx, col] != 0:
            user_anime_history_dict[col] = user_anime_history_df.at[user_idx, col]
    return user_anime_history_dict


def get_user_idx(user_id, user_df):
    """Returns the index of the user from a df with user data.

    Args:
        user_id: MyAnimeList user ID.
        user_df: Either the user_anime_history_df or the user_score_df.
    """
    return user_df[user_df['user_id'] == user_id].index[0]


def get_collab_filt_recs(user_id, dist_matrix,
                         anime_titles, user_score_df, num_recs=10):
    """Returns the collaborative-filtering recommendations for a user_id.

    Args:
        user_id: MyAnimeList user ID.
        dist_matrix: Pairwise distance matrix between user embeddings
        and anime embeddings.
        anime_titles: List of anime titles considered in recommender system.
        user_score_df: User-rating matrix.
        num_recs: Number of recommendations.
    """
    user_idx = get_user_idx(user_id, user_score_df)
    # Get back recommendations without any recommendations that user has in their
    # anime list
    # Limiting to 50 for Flask to speed up recommendations
    # But may run into situations where don't get back 10 recommendations if
    # user has already watched a lot of the top 1000 anime
    recs = [anime_titles[idx] for idx in dist_matrix[user_idx].argsort()[:50]
            if anime_titles[idx] not in
            get_user_scores(user_id, user_score_df)][:num_recs]
    return recs


def get_content_filt_recs(user_id, dist_matrix,
                          anime_titles, user_anime_history_df, num_recs=10):
    """Return the content-based filtering recommendations for a user_id.

    Args:
        user_id: MyAnimeList user ID.
        dist_matrix: Pairwise distance matrix between user embeddings
        and anime embeddings.
        anime_titles: List of anime titles considered in recommender system.
        user_anime_history_df: DataFrame of user's anime viewing history where
        the entries 1 = watched and 0 = not watched.
        num_recs: Number of recommendations.
    """
    user_idx = get_user_idx(user_id, user_anime_history_df)
    # Get back recommendations without any recommendations that user has in their
    # anime list
    # Limiting to 50 for Flask to speed up recommendations
    # But may run into situations where don't get back 10 recommendations if
    # user has already watched a lot of the top 1000 anime
    recs = [anime_titles[idx] for idx in dist_matrix[user_idx].argsort()[:50]
            if anime_titles[idx] not in
            get_user_anime_history(user_id, user_anime_history_df)][:num_recs]
    return recs

def combine_double_recs(recs_df):
    """Returns recs_df after double anime recommendations have been combined.

    Args:
        recs_df = DataFrame of recommendations from content-based and collaborative
        filtering.
    """
    # Get list of all double anime recommendations (i.e. where collaborative and content-based
    # filtering recommend the same anime)
    double_anime_recs = recs_df['anime_rec'].value_counts() \
        [recs_df['anime_rec'].value_counts() > 1].index.tolist()

    # Convert the rec_type for each duplicate entry to 'both content/collab'
    for anime in double_anime_recs:
        recs_df.loc[recs_df['anime_rec'] == anime, 'rec_type'] = 'both content/collab'

    # Use groupby to combine the entries and scores for the double recommendations.
    recs_df = recs_df.groupby(['user_id', 'anime_rec', 'rec_type', 'original_rank'], \
        as_index=False).agg({'base_score': sum, 'weighted_score': sum})

    return recs_df

def recommend(user_id, user_anime_cosine_distances_content,
              user_anime_cosine_distances_collab, user_score_df,
              user_anime_history_df, anime_titles, collab_weight=1,
              num_recs=10):
    """Makes anime recommendations based on user_id and scoring logic.

    Args:
        user_id: MyAnimeList user ID.
        user_anime_cosine_distances_content: Using content-based filtering embeddings,
        pairwise cosine distance matrix between user embeddings and anime embeddings.
        user_anime_cosine_distances_collab: Using collaborative filtering embeddings,
        pairwise cosine distance matrix between user embeddings and anime embeddings.
        user_score_df: User-rating matrix.
        user_anime_history_df: DataFrame of user's anime viewing history where
        the entries 1 = watched and 0 = not watched.
        anime_titles: List of anime titles considered in recommender system.
        collab_weight = Weight applied to collaborative filtering scores.
        num_recs: Number of recommendations.
    Returns:
        recs: Recommendations as a list of anime where the list length equals
        the parameter num_recs. List is sorted with top recommendations first.
        recs_df: DataFrame of recommendations with details.
    """
    rec_dicts = []
    collab_recs = get_collab_filt_recs(user_id, user_anime_cosine_distances_collab,
                                       anime_titles, user_score_df)
    content_recs = get_content_filt_recs(user_id, user_anime_cosine_distances_content,
                                         anime_titles, user_anime_history_df)
    for idx, (collab_rec, content_rec) in enumerate(zip(collab_recs, content_recs)):
        rec_dict_collab = {
            'user_id': user_id,
            'anime_rec': collab_rec,
            'rec_type': 'collab',
            'original_rank': idx+1,
            'base_score': 10-idx,
            'weighted_score': (10-idx)*collab_weight
        }
        rec_dict_content = {
            'user_id': user_id,
            'anime_rec': content_rec,
            'rec_type': 'content',
            'original_rank': idx+1,
            'base_score': 10-idx,
            'weighted_score': 10-idx
        }
        rec_dicts = rec_dicts + [rec_dict_collab] + [rec_dict_content]

    recs_df = pd.DataFrame(rec_dicts)
    recs_df = combine_double_recs(recs_df)
    # Sort recommendations by weighted_score and rec_type (so collab goes before content)
    recs_df = recs_df.sort_values(['weighted_score', 'rec_type'], \
        ascending=[False, True], ignore_index=True)
    recs = recs_df.iloc[0:num_recs]['anime_rec'].values.tolist()
    return recs, recs_df
