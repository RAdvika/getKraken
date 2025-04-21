"""Generate database for commits & issues"""

import json
import os.path
import re
import requests
from typing import Tuple, List
from io import StringIO

import kagglehub
import pandas as pd

from html.parser import HTMLParser
from dotenv import dotenv_values
from github import Github, GithubException, Auth


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
DB_WRITE = './db/n_sample_data.json'
SAMPLE_SIZE = 250

GRAPHQL_URL = 'https://api.github.com/graphql'

GRAPHQL_ISSUE_HEADER = {
    'Authorization': f'bearer {env['GITHUB_TOKEN']}',
    'Accept': 'application/vnd.github.github+json'
}
GRAPHQL_COMMIT_HEADER = {
    'Authorization': f'bearer {env['GITHUB_TOKEN']}',
    'Content-Type': 'application/json'
}
GRAPHQL_DIFF_HEADER = {
    'Authorization': f'bearer {env['GITHUB_TOKEN']}',
    'Accept': 'application/vnd.github.v3.diff'
}

GRAPHQL_ISSUE_QUERY = """
query($owner: String!, $repo: String!) {
  repository(owner: $owner, name: $repo) {
    issues(first: 100, states: CLOSED, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        body 
        title
        url
      }
    }
  }
}
"""
GRAPHQL_COMMITS_QUERY = """
query ($owner: String!, $repo: String!, $branch: String!, $after: String) {
  repository(owner: $owner, name: $repo) {
    ref(qualifiedName: $branch) {
      target {
        ... on Commit {
          history(first: 100, after: $after) {
            pageInfo {
              hasNextPage
              endCursor
            }
            edges {
              node {
                messageHeadline
                messageBody
                oid
                url 
              }
            }
          }
        }
      }
    }
  }
}
"""

GIT_DIFF_PAT = re.compile(r"^diff --git .*$", re.MULTILINE)
GIT_DIFF_FILE_A = re.compile(r"a/\S+")
GIT_DIFF_FILE_B = re.compile(r"b/\S+")
COMMIT_TITLE_FILTER = r"(^Update [.\S]*$)|(^Merge branch '.*' into [\S]*$)"

README_FILTER_LIST = [
    r'[dD]atabase of \S',
    r'[cC]ollection of \S',
    r'for beginners'
]
README_FILTER = re.compile('|'.join(README_FILTER_LIST))

REPO_FILTER_LIST = [
    r"[tT]utorial",
    "octocat",
    "vscode",
    r"[rR]oadmap",
    r"[cC]hallenges",
    r"[iI]ntro.[tT]o",
    r"[wW]iki"
]
REPO_FILTER = re.compile('|'.join(REPO_FILTER_LIST))


def get_diffs(diff_raw: str):
    files = GIT_DIFF_PAT.findall(diff_raw)
    diffs = GIT_DIFF_PAT.split(diff_raw)[1:]
    assert len(files) == len(diffs)

    return [
        {
            'filepath':
                {
                    'prev': GIT_DIFF_FILE_A.findall(f),
                    'curr': GIT_DIFF_FILE_B.findall(f)
                },
            'diff': d
        }
        for f, d in zip(files, diffs)]


def get_issues(owner: str, repo_name: str):
    issue_vars = {
        'owner': owner,
        'repo': repo_name
    }
    issues_data = requests.post(GRAPHQL_URL,
                                json={'query': GRAPHQL_ISSUE_QUERY, 'variables': issue_vars},
                                headers=GRAPHQL_ISSUE_HEADER).json()
    issues_from_request = issues_data['data']['repository']['issues']['nodes']

    issues = []
    for issue in issues_from_request:
        print(f'\tissue #{issue['number']}: {issue['title']} ({issue['url']})')
        issues.append({
            'title': issue['title'],
            'body': issue['body'],
            'url': issue['url']
        })
    return issues


def get_commits(owner: str, repo_name: str, default_branch: str):
    total_commits = []
    commits, after = get_commit_page(owner, repo_name, default_branch)
    total_commits.extend(commits)
    # add at least 250 commits per repo if possible
    while len(total_commits) < 250 and after:
        commits, after = get_commit_page(owner, repo_name, default_branch, after=after)
        total_commits.extend(commits)
    return total_commits


def get_commit_page(owner: str, repo_name: str, default_branch: str, after: str = None) -> Tuple[List, str]:
    commit_vars = {
        'owner': owner,
        'repo': repo_name,
        'branch': default_branch,
        'after': after
    }
    commit_data = requests.post(GRAPHQL_URL,
                                json={'query': GRAPHQL_COMMITS_QUERY, 'variables': commit_vars},
                                headers=GRAPHQL_ISSUE_HEADER).json()
    history = commit_data['data']['repository']['ref']['target']['history']

    commits = []
    for edge in history['edges']:
        node = edge['node']
        if re.match(COMMIT_TITLE_FILTER, node['messageHeadline']):
            continue

        print(f'\tcommit {node['oid']} {node['messageHeadline']}')
        commit = {
            'sha': node['oid'],
            'header': node['messageHeadline'],
            'body': node['messageBody'],
            'url': node['url']
        }

        commits.append(commit)

    return commits, history['pageInfo']['endCursor'] if history['pageInfo']['hasNextPage'] else None


def main():
    # download latest version
    path = kagglehub.dataset_download('donbarbos/github-repos')
    path = os.path.join(path, 'repositories.csv')

    print('Path to dataset files:', path)

    df = pd.read_csv(path)
    df = df.dropna()  # drop all rows with NaN values
    df = df[df['Stars'] > 100]  # drop all repos with less than 100 stars
    df = df[df['Issues'] > 25]  # drop all repos with less than 25 issues open
    df = df.drop_duplicates()  # drop dupes

    print(df)
    print(df.shape)

    try:
        with open(DB_PATH, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f'error: {e}')

        auth = Auth.Token(env['GITHUB_TOKEN'])
        g = Github(auth=auth)

        dataset = {}
        for r in df['URL']:
            if len(dataset) >= SAMPLE_SIZE:
                break

            s = r.split('/')
            r = f'{s[-2]}/{s[-1]}'
            owner = r.split('/')[0]
            repo_name = r.split('/')[1]

            print(f'({len(dataset)}/{SAMPLE_SIZE}) adding repo {r}')

            try:
                if REPO_FILTER.search(r):
                    print('\tskip for repo filter')
                    continue

                repo_url = g.get_repo(r)
                if 'python' not in repo_url.get_topics():
                    print('\tskip for not python')
                    continue

                raw_readme = str(repo_url.get_readme().decoded_content)
                p = HTMLStripper()
                p.feed(raw_readme)
                readme = p.get_data()

                if README_FILTER.search(readme):
                    print('\tskip for readme filter')
                    continue

                print('\tadding issues...')
                issues = get_issues(owner, repo_name)

                print('\tadding commits...')
                commits = get_commits(owner, repo_name, repo_url.default_branch)

                repo_url = df[df['URL'] == f'https://github.com/{r}']
                issues_count = repo_url['Issues'].iloc[0]
                desc = repo_url['Description'].iloc[0]

                dataset[r] = {
                    'readme': readme,
                    'short desc': str(desc),
                    'forks count': repo_url.forks_count,
                    'stars count': repo_url.stargazers_count,
                    'issues count': int(issues_count),
                    'issues': issues,
                    'commits': commits
                }
            except GithubException as e:
                print(e)
                continue

        g.close()

        with open(DB_WRITE, 'w') as f:
            json.dump(dataset, f)


if __name__ == "__main__":
    main()
