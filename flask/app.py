import pickle
from flask import Flask, redirect, url_for, request, render_template
from recommendation import recommender as rec
app = Flask(__name__)

with open('../pickles/rec_data.pkl', 'rb') as f:
    user_anime_cosine_distances_content, user_anime_cosine_distances_collab, \
        user_score_df, user_anime_history_df, anime_titles = pickle.load(f)

# To pass a variable into my request function, I need to put it into the URL
@app.route('/recommendation/<user_id>/<adventurous_level>', methods=['POST', 'GET'])
def recommendation(user_id, adventurous_level):
    if request.method == 'POST':
        # Use request.form for POST requests and request.args for GET requests
        return redirect(url_for('recommendation',
                                user_id=request.form.get('user_id'),
                                adventurous_level=request.form.get('adventurous_level')))

    recs, _ = rec.recommend(user_id, user_anime_cosine_distances_content,
                            user_anime_cosine_distances_collab, user_score_df,
                            user_anime_history_df, anime_titles,
                            collab_weight=float(adventurous_level))
    return render_template('recommendation.html', recs=recs, user_id=user_id,
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
