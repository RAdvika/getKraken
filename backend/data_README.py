"""Generate database for README files"""

import json
import os.path
from io import StringIO

import kagglehub
import pandas as pd

from dotenv import dotenv_values
from github import Github, GithubException
from github import Auth
from html.parser import HTMLParser


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


env = dotenv_values('./env/.env')

DB_PATH = './db/sample_data.json'
SAMPLE_SIZE = 2500

# download latest version
path = kagglehub.dataset_download("donbarbos/github-repos")
path = os.path.join(path, 'repositories.csv')

print("Path to dataset files:", path)

df = pd.read_csv(path)
df = df.dropna()  # drop all rows with NaN values
df = df[df['Stars'] > 100]  # drop all repos with less than 100 stars
df = df[df['Issues'] > 25]  # drop all repos with less than 25 issues open
df = df.drop_duplicates()  # drop dupes

print(df)
print(df.shape)

top_10_langs = {'javascript', 'python', 'java', 'typescript', 'csharp', 'cpp', 'php', 'shell', 'c', 'ruby'}

try:
    with open(DB_PATH, 'r') as f:
        dataset = json.loads(f)

except:
    auth = Auth.Token(env['GITHUB_TOKEN'])
    g = Github(auth=auth)

    dataset = {}
    for l in top_10_langs:
        dataset[l] = {}

    # get first 2500 repos for sampling (there are over 200k actual repos we can use)
    for i, r in enumerate(df['URL'][:SAMPLE_SIZE]):
        s = r.split('/')
        r = f'{s[-2]}/{s[-1]}'
        print(f'({i + 1}/{SAMPLE_SIZE}) adding repo {r}')

        bag = {}
        try:
            repo = g.get_repo(r)

            labels = set([l.name for l in repo.get_labels()])
            langs = labels.intersection(top_10_langs)
            # if not part of top 10 langs, continue
            if len(langs) == 0:
                continue

            raw_txt = str(repo.get_readme().decoded_content)
            p = HTMLStripper()
            p.feed(raw_txt)

            bag['repository'] = r
            bag['readme'] = p.get_data()

            for l in langs:
                dataset[l][r] = bag

        except GithubException as e:
            print(e)
            continue

    g.close()
    with open(DB_PATH, 'w') as f:
        json.dump(dataset, f)

for l in top_10_langs:
    print(f'{l}: {len(dataset[l])} repos')
