import pickle
from flask import Flask, redirect, url_for, request, render_template
from recommendation.recommender import recommend
app = Flask(__name__)

with open('../pickles/rec_data.pkl', 'rb') as f:
    user_anime_cosine_distances_content, user_anime_cosine_distances_collab, \
        user_score_df, user_anime_history_df, anime_titles = pickle.load(f)

with open('../pickles/cleaned_top_anime_data_1000_df.pkl', 'rb') as f:
    top_anime_df = pickle.load(f)

# To pass a variable into my request function, I need to put it into the URL
@app.route('/recommendation/<user_id>/<adventurous_level>', methods=['POST', 'GET'])
def recommendation(user_id, adventurous_level):
    if request.method == 'POST':
        # Use request.form for POST requests and request.args for GET requests
        return redirect(url_for('recommendation',
                                user_id=request.form.get('user_id'),
                                adventurous_level=request.form.get('adventurous_level')))

    recs, _ = recommend(user_id, user_anime_cosine_distances_content,
                        user_anime_cosine_distances_collab, user_score_df,
                        user_anime_history_df, anime_titles,
                        collab_weight=float(adventurous_level))

    recs_dicts = [
        {'anime_title': anime_title,
         'url': top_anime_df[top_anime_df['title_main'] == anime_title]['url'].values[0],
         'image_url': top_anime_df[top_anime_df['title_main'] == anime_title]['image_url'].values[0]}
        for anime_title in recs
    ]

    # Need to create a list of dicts to store anime_title and image_url
    # and then pass list of dicts into render_template to display anime title and image
    return render_template('recommendation.html', recs=recs_dicts, user_id=user_id,
                           adventurous_level=adventurous_level)

@app.route('/', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        # Use request.form for POST requests and request.args for GET requests
        return redirect(url_for('recommendation',
                                user_id=request.form.get('user_id'),
                                adventurous_level=request.form.get('adventurous_level')))
    return render_template('index.html')

if __name__ == '__main__':
    # Allows me to make changes to app without restarting server
    app.run(debug=True)
