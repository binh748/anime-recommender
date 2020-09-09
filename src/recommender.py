"""This module contains functions for the anime recommender system."""


def get_user_scores(user_id, user_score_df):
    """Returns dictionary of all non-zero user scores, including
    user/animelist metadata."""
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
    """
    user_idx = get_user_idx(user_id, user_anime_history_df)
    user_anime_history_dict = {}
    for col in user_anime_history_df[user_anime_history_df['user_id'] == user_id]:
        if user_anime_history_df.at[user_idx, col] != 0:
            user_anime_history_dict[col] = user_anime_history_df.at[user_idx, col]
    return user_anime_history_dict


def get_user_idx(user_id, user_df):
    """Returns the index of the user from a df with user data."""
    return user_df[user_df['user_id'] == user_id].index[0]


# May need to change this function to make it quicker
def get_collab_filt_recs(user_id, dist_matrix,
                         anime_titles, user_score_df, num_recs=10):
    """Return the collaborative-filtering recommendations for a user_id."""
    user_idx = get_user_idx(user_id, user_score_df)
    # Get back recommendations without any recommendations that user has in their
    # anime list
    recs = [anime_titles[idx] for idx in dist_matrix[user_idx].argsort()
            if anime_titles[idx] not in
            get_user_scores(user_id, user_score_df)][:num_recs]
    return recs


# May need to change this function to make it quicker
def get_content_filt_recs(user_id, dist_matrix,
                          anime_titles, user_anime_history_df, num_recs=10):
    """Return the content-based filtering recommendations for a user_id."""
    user_idx = get_user_idx(user_id, user_anime_history_df)
    # Get back recommendations without any recommendations that user has in their
    # anime list
    recs = [anime_titles[idx] for idx in dist_matrix[user_idx].argsort()
            if anime_titles[idx] not in
            get_user_anime_history(user_id, user_anime_history_df)][:num_recs]
    return recs
