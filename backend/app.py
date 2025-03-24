import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
# from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from helpers import ranker
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))



# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, "db/sample_data.json")
json_file_path = os.path.join(current_directory, "db/sample_data.json")




top_10_langs = {'javascript', 'python', 'java', 'typescript', 'csharp', 'cpp', 'php', 'shell', 'c', 'ruby'}
with open(json_file_path, 'r') as file:
    data = json.load(file)


app = Flask(__name__)
CORS(app)


# Sample search for repos in specific lang
def sample_search(query, lang):
    input_json = dict()

    for l in lang:
        lang_json = data.get(l)
        if lang_json:
            input_json[l] = lang_json
    
    return ranker(input_json, query)[:5].to_json(orient='records')

    


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/repo")
def repo_search():
    repo = request.args.get("repo")
    lang_str = request.args.get("lang")
    lang = lang_str.split(',') if lang_str else []

    return sample_search(repo, lang)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

