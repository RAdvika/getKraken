import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from typing import List, Tuple
from helpers.ranker import Ranker
import numpy as np
DEMO_MODE = True


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

def format_json(ranked: tuple[int, int, int, float], total: int):
    result_json = {
        'total': total,
        'results':  []
    }
    for idx, cm_idx, is_idx, sim in ranked:
        repo = ranker.repositories[idx]

        commit = repo.commits[cm_idx]
        #idk why this is causing out of index error 
        # issue = repo.issues[is_idx]


        if isinstance(repo.readme, bytes):
            readme_text = repo.readme.decode('utf-8')
        else:
            readme_text = repo.readme[2:]

        commit_json  = {
            'title' : commit.header,
            'body' : commit.body,
            'url' : commit.url
        }

        # issue_json = {
        #     'title' : issue.title,
        #     'body' : issue.body,
        #     'url' : issue.url
        # }

        repo_json = {
            'repo_name': repo.repo_name,
            'language': 'python',
            'short_desc' : repo.short_desc,
            'readme_raw': readme_text,
            'similarity': sim*100,
            'stars' : repo.stars_count,
            'forks' : repo.forks_count,
            'issues' : repo.issues_count,
            'commit' : commit_json,
            # 'issues_info' : issue_json
            }

        result_json['results'].append(repo_json)

    return result_json

def meta_data_rank(a = 0.7,
                   b = 0.1,
                   y = 0.2, 
                   scores: list[tuple[int, float]] = None)  -> list[tuple[int, float]]:
    
    if scores is None or len(scores) == 0:
        return []
    
    results = []

    for idx, sim in scores:
        stars = ranker.repositories[idx].stars_count
        forks = ranker.repositories[idx].forks_count

        stars_score = np.log(1 + stars)
        forks_score = np.log(1 + forks)
        rank_score  = a*sim + b*stars_score + y*forks_score 

        results.append(idx,sim,rank_score)

    ranked_result = sorted(results, key=lambda x: x[2], reverse=True)

    return [(idx, sim) for idx, sim, _ in ranked_result]

    


@app.route("/")
def home():
    return render_template('base.html', is_demo_mode=("true" if DEMO_MODE else "false"))


@app.route("/repo")
def repo_search():
    repo = request.args.get("repo")
    lang_str = request.args.get("lang")
    page = int(request.args.get("page"))
    per_page = int(request.args.get("per_page"))
    lang = lang_str.split(',') if lang_str else []

    key = f"{repo}_{'_'.join(sorted(lang))}"
    ranked = search(key, repo)

    start = (page - 1) * per_page
    end = start + per_page
    paginated = ranked[start:end]

    return jsonify(format_json(paginated, len(ranked)))



if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

