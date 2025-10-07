from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

# ====== Direct Google Drive Download Links ======
SIMILARITY_URL = "https://drive.google.com/uc?export=download&id=1EK0R1SSmjUR4OW26DSbREBlideZYnHA0"
COURSES_URL = "https://drive.google.com/uc?export=download&id=1COrh6Y3g8XLVwc9LrnIX7hOL9cE_gXr2"
COURSE_LIST_URL = "https://drive.google.com/uc?export=download&id=1kWtYzHBPQwa8YeFqTY45ZNE07t4c_xHA"

# ====== Detect if running on Vercel ======
ON_VERCEL = os.environ.get("VERCEL") == "1"

# ====== Model directory (use /tmp on Vercel, models/ locally) ======
MODEL_DIR = "/tmp/models" if ON_VERCEL else "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")
COURSES_PATH = os.path.join(MODEL_DIR, "courses.pkl")
COURSE_LIST_PATH = os.path.join(MODEL_DIR, "course_list.pkl")

# ====== Download helper ======
def download_model(url, path):
    if not os.path.exists(path):
        print(f"📥 Downloading model: {path}")
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed to download {path}: {e}")

# ====== Ensure models exist ======
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

# ====== Flask Routes ======
@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_courses = []
    selected_course = None
    if request.method == 'POST':
        selected_course = request.form['course_name']
        recommended_courses = recommend(selected_course)
    return render_template('index.html', courses=course_names, recommendations=recommended_courses, selected_course=selected_course)

# ====== Run locally ======
if __name__ == "__main__":
    app.run(debug=True)
