from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import requests
import os
from vercel import Vercel

app = Flask(__name__)

# ====== Model URLs ======
SIMILARITY_URL = "https://example.com/similarity.pkl"
COURSES_URL = "https://example.com/courses.pkl"
COURSE_LIST_URL = "https://example.com/course_list.pkl"

# ====== Local paths in /tmp ======
SIMILARITY_PATH = "/tmp/similarity.pkl"
COURSES_PATH = "/tmp/courses.pkl"
COURSE_LIST_PATH = "/tmp/course_list.pkl"

# ====== Download models ======
def download_model(url, path):
    if not os.path.exists(path):
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

download_model(SIMILARITY_URL, SIMILARITY_PATH)
download_model(COURSES_URL, COURSES_PATH)
download_model(COURSE_LIST_URL, COURSE_LIST_PATH)

# ====== Load models ======
similarity = pickle.load(open(SIMILARITY_PATH, 'rb'))
courses_df = pickle.load(open(COURSES_PATH, 'rb'))
course_list_dicts = pickle.load(open(COURSE_LIST_PATH, 'rb'))

course_names = courses_df['course_name'].values.tolist()
course_url_dict = courses_df.set_index('course_name')['course_url'].to_dict()

# ====== Recommendation Function ======
def recommend(course_name):
    if course_name not in courses_df['course_name'].values:
        return []

    index = courses_df[courses_df['course_name'] == course_name].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_courses = []
    for i in distances[1:7]:
        recommended_name = courses_df.iloc[i[0]].course_name
        recommended_url = courses_df.iloc[i[0]].course_url
        recommended_courses.append({'name': recommended_name, 'url': recommended_url})
    return recommended_courses

# ====== Routes ======
@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_courses = []
    selected_course = None
    if request.method == 'POST':
        selected_course = request.form['course_name']
        recommended_courses = recommend(selected_course)
    return render_template('index.html', courses=course_names, recommendations=recommended_courses, selected_course=selected_course)

# ====== Wrap for Vercel ======
app = Vercel(app)
