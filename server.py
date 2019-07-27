from flask import Flask, render_template, request, abort
import io

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 10

@app.errorhandler(404)
def page_not_found(e):
    return render_template("bad.html"), 404

@app.route('/')
def home_page():
    return render_template("home.html")

@app.route('/about')
def about_page():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)