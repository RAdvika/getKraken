import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from typing import List, Tuple
from helpers.ranker import Ranker, Result
import numpy as np
from functools import lru_cache

DEMO_MODE = True


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, '..', 'db', "sample_data.json")
json_file_path = os.path.join(current_directory, "sample_data.json")


top_10_langs = {'javascript', 'python', 'java', 'typescript', 'csharp', 'cpp', 'php', 'shell', 'c', 'ruby'}
with open(json_file_path, 'r') as file:
    data = json.load(file)

ranker = Ranker(data)

app = Flask(__name__)
# cache = {}
CORS(app)



def format_json(ranked: list[Result], total: int):
    result_json = {
        'total': total,
        'results':  []
    }
    for result in ranked:
        idx = result.repo_idx
        cm_idx = result.commit_idx
        is_idx = result.issue_idx
        coocmatrix_feature = result.coocmatrix_feature
        hidden_d = result.hidden_d
        sim = result.sim_score

        repo = ranker.repositories[idx]

        if isinstance(repo.readme, bytes):
            readme_text = repo.readme.decode('utf-8')
        else:
            readme_text = repo.readme[2:]

        if cm_idx >= 0:
            commit = repo.commits[cm_idx]
            commit_json  = {
                'title' : commit.header,
                'url' : commit.url,
                'author' :  commit.author
            }
        else:
            commit_json  = {
                'title' : '',
                'url' : '',
                'author' :  ''
            }
        if is_idx >= 0:
            issue = repo.issues[is_idx]
            issue_json = {
                'title' : issue.title,
                'url' : issue.url
            }
        else:
            issue_json = {
                'title' : '',
                'url' : ''
            }            

        repo_json = {
            'repo_name': repo.repo_name,
            'language': 'python',
            'coocmatrix': coocmatrix_feature,
            'svd_features': hidden_d,
            'short_desc' : repo.short_desc,
            'readme_raw': readme_text,
            'similarity': sim*100,
            'stars' : repo.stars_count,
            'forks' : repo.forks_count,
            'issues' : repo.issues_count,
            'commit' : commit_json,
            'issues_info' : issue_json
            }

        result_json['results'].append(repo_json)

    return result_json

def meta_data_rank(a = 0.7,
                   b = 0.1,
                   y = 0.2, 
                   repo_list: list[Result] = None)  -> list[Result]:
    
    max_star = max_fork = 1

    for result in repo_list:
        max_star = max(ranker.repositories[result.repo_idx].stars_count, max_star)
        max_fork = max(ranker.repositories[result.repo_idx].forks_count, max_fork)


    max_stars_log = np.log(max_star)
    max_fork_log = np.log(max_fork)

    result_with_scores = []

    for result in repo_list:
        stars = ranker.repositories[result.repo_idx].stars_count
        forks = ranker.repositories[result.repo_idx].forks_count
        sim = result.sim_score  


        stars_log = np.log1p(stars)
        forks_log = np.log1p(forks)
        stars_norm = stars_log / max_stars_log
        forks_norm = forks_log / max_fork_log
        score = a * sim + b * stars_norm + y * forks_norm
        result_with_scores.append((result, score))


    ranked = sorted(result_with_scores, key=lambda x: x[1], reverse=True)

    return [repo for repo, _ in ranked]

@lru_cache(maxsize=50)
def search(key: str, query: str) -> list[Result]:
    ranked = ranker.rank(query)
    return meta_data_rank(repo_list=ranked)


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
    ranked_result =  search(key, repo)

    start = (page - 1) * per_page
    end = start + per_page
    paginated = ranked_result[start:end]

    return jsonify(format_json(paginated, len(ranked_result)))



if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

