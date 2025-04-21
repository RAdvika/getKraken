import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from typing import List, Tuple
from helpers.ranker import Ranker

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, '..', 'db', "sample_data.json")

top_10_langs = {'javascript', 'python', 'java', 'typescript', 'csharp', 'cpp', 'php', 'shell', 'c', 'ruby'}
with open(json_file_path, 'r') as file:
    data = json.load(file)

ranker = Ranker(data)

app = Flask(__name__)
cache = {}
CORS(app)


def search(key: str, query: str) -> list[tuple[int, float]]:
    if key in cache:
        ranked_results = cache[key]
    else:
        ranked_results = ranker.rank(query)
        cache[key] = ranked_results

    return ranked_results

def format_json(ranked: tuple[int, float]):
    result_json = {
        'total': len(ranked),
        'results':  []
    }
    for idx, sim in ranked:
        repo = ranker.repositories[idx]

        repo_json = {
            'repo_name': repo.repo_name,
            'language': 'python',
            'readme_raw': repo.readme,
            'similarity': sim,
            'stars' : repo.stars_count,
            'forks' : repo.forks_count,
            'issues' : repo.issues_count
            }

        result_json['results'].append(repo_json)

    return result_json


    


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/repo")
def repo_search():
    repo = request.args.get("repo")
    lang_str = request.args.get("lang")
    page = int(request.args.get("page"))
    per_page = int(request.args.get("per_page"))
    lang = lang_str.split(',') if lang_str else []

    key = f"{repo}_{'_'.join(sorted(lang))}"
    ranked = search(key, repo)

    start = page * per_page
    end = start + per_page
    paginated = ranked[start:end]

    return jsonify(format_json(paginated))



    # start = (page - 1) * per_page
    # end = start + per_page
    # paginated = ranked_results.iloc[start:end]

    # paginated['similarity'] = paginated['similarity'] * 100

    # total =  len(ranked_results)
    # return {
    #     "total": total,
    #     "results" :paginated.to_json(orient='records')
    # }


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

