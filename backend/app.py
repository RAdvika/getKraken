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

# # Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, '..', 'db', "sample_data.json")

json_file_path = os.path.join(current_directory, "sample_data.json")


top_10_langs = {'javascript', 'python', 'java', 'typescript', 'csharp', 'cpp', 'php', 'shell', 'c', 'ruby'}
with open(json_file_path, 'r') as file:
    data = json.load(file)

ranker = Ranker(data)

app = Flask(__name__)
cache = {}
CORS(app)



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
                   repo_list: list[tuple[int, int, int, float]] = None)  -> list[tuple[int, int, int, float]]:
    
    if not repo_list:
        return []

    repo_array = np.array(repo_list)
    indices = repo_array[:, 0].astype(int)
    sims = repo_array[:, 3].astype(float)

    stars = np.array([ranker.repositories[idx].stars_count for idx in indices])
    forks = np.array([ranker.repositories[idx].forks_count for idx in indices])

    stars_log = np.log1p(stars)
    forks_log = np.log1p(forks)

    max_stars_log = np.max(stars_log) or 1
    max_fork_log = np.max(stars_log) or 1

    stars_norm = stars_log / max_stars_log
    forks_norm = forks_log / max_fork_log

    scores = a * sims + b * stars_norm + y * forks_norm

    result_with_scores = list(zip(repo_list, scores))

    ranked = sorted(result_with_scores, key=lambda x: x[1], reverse=True)

    return [repo for repo, _ in ranked]


def search(key: str, query: str) -> list[tuple[int, int, int, float]]:
    if key in cache:
        ranked_results = cache[key]
    else:
        search_result = ranker.rank(query)
        ranked_results = meta_data_rank(repo_list=search_result)
        cache[key] = ranked_results
    return ranked_results    


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

