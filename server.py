from flask import Flask, render_template, request, abort
import io
from recommender import recommender as rec
from collections import defaultdict

recommender = rec()
recommender.populate_user_ratings("data/ratings.csv")
recommender.populate_movie_names("data/movies.csv")
recommender.populate_links("data/links.csv")
recommender.populate_movie_url_images("data/imgURLs.csv")
movie_dict = defaultdict(str)
url_dict = defaultdict(str)
user_set = set()

for row in recommender.moviesDF.itertuples():
    movie_dict[int(row[1])] = row[2]
movies = sorted(movie_dict.items(), key = lambda kv: kv[1])
for row in recommender.ratingsDF.itertuples():
    user_set.add(row[1])
users = sorted(user_set)
for row in recommender.imgURLsDF.itertuples():
    url_dict[row[1]] = row[2]

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 10

@app.errorhandler(404)
def page_not_found(e):
    return render_template("bad.html"), 404

@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        result = request.form
        who = int(result['user'])
        try:
            what = int(result['movie'])
            name = movie_dict[what]
            link = url_dict[what]
        except Exception:
            what = int(result['movie'])
            name = movie_dict[what]
            link = "static/img/none.png"
        if link == "missing":
            link = "static/img/none.png" 
        print(link)
        return render_template("result.html", movie=name, url=link, user=who)
    return render_template("home.html", users=users, movies=movies)

@app.route('/about')
def about_page():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)