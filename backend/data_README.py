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

try:
    with open('sample_data.json', 'r') as f:
        dataset = json.loads(f.read())
        # TODO: load data in

except:
    auth = Auth.Token(env['GITHUB_TOKEN'])
    g = Github(auth=auth)

    dataset = {}
    # get first 1000 repos for sampling (there are over 200k actual repos we can use)
    for r in df['URL'][:1000]:
        s = r.split('/')
        r = f'{s[-2]}/{s[-1]}'
        print(f'>> adding repo {r}')

        bag = {}
        try:
            repo = g.get_repo(r)

            labels = repo.get_labels()
            # drop repos with no labels
            if labels.totalCount == 0:
                continue
            labels = [l.name for l in labels]

            raw_txt = str(repo.get_readme().decoded_content)
            p = HTMLStripper()
            p.feed(raw_txt)

            bag['repository'] = r
            bag['readme'] = p.get_data()
            bag['labels'] = labels
            dataset[r] = bag
        except GithubException as e:
            print(e)
            continue

    g.close()
    with open('sample_data.json', 'w') as f:
        json.dump(dataset, f)

print(f'{len(dataset)} repos added')

# TODO: process data here
