import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from backend.helpers.parse_query import preprocess_query
from .parse_query import preprocess_query
from typing import Tuple



class Issue:
    def __init__(self, title : str, body: str, url: str):
        self.title = title
        self.body = body
        self.body_len = len(body)
        self.url = url


class Commit:
    def __init__(self, sha : str, header: str, body:  str,  url: str):
        self.sha = sha
        self.header = header
        self.body = body
        self.body_len = len(body)
        self.url = url

class Repository:
    def __init__(self, repo_name : str,
                 readme : str, 
                 short_desc : str, 
                 forks_count : int, 
                 stars_count : int, 
                 issues_count : int, 
                 issues , 
                 commits):
        self.repo_name = repo_name
        self.readme = readme
        self.readme_len = len(readme)
        self.short_desc = short_desc
        self.forks_count = forks_count
        self.stars_count = stars_count
        self.issues_count = issues_count
        self.issues = issues  
        self.commits = commits  
        self.commits_count = len(commits)

class Ranker:
    def __init__(self, json_data: dict):
        self.repositories = self.parse_data(json_data)

    def parse_data(self, json_data) -> list[Repository]:
        repositories = []
        for repo_name, repo_data in json_data.items():
            issues = [Issue(issue['title'], issue['body'], issue['url']) for issue in repo_data.get('issues', [])]
            commits = [Commit(commit['sha'], commit['header'], commit['body'], commit['url']) for commit in repo_data.get('commits', [])]

            repo = Repository(
                repo_name=repo_name,
                readme=repo_data.get('readme', ''),
                short_desc=repo_data.get('short desc', ''),
                forks_count=repo_data.get('forks count', 0),
                stars_count=repo_data.get('stars count', 0),
                issues_count=repo_data.get('issues count', 0),
                issues=issues,
                commits=commits
            )
            repositories.append(repo)
        return repositories
    
    def max_commit_score(self, query_vector, vectorizer, repo):
        max_score =  0.0
        max_index = -1

        for idx, commit in  enumerate(repo.commits):
            commit_text = commit.body
            commit_vector = vectorizer.transform([commit_text])
            similarity = cosine_similarity(query_vector, commit_vector)[0, 0]

            if similarity > max_score:
                max_score = similarity
                max_index = idx

        return  max_index, max_score
    
    def max_issue_score(self, query_vector, vectorizer, repo):
        max_score =  0.0
        max_index = -1

        for idx, issue in  enumerate(repo.issues):
            issue_text = issue.body
            issue_vector = vectorizer.transform([issue_text])
            similarity = cosine_similarity(query_vector, issue_vector)[0, 0]

            if similarity > max_score:
                max_score = similarity
                max_index = idx

        return  max_index, max_score



    def weighted_score(self, 
                       repo:  Repository, 
                       readme_score: float,
                       commit_res: Tuple[int, float], 
                       issue_res: Tuple[int, float]):
        cm_idx, cm_score  = commit_res
        is_idx,  is_score  = issue_res


        rm_len = len(repo.readme)
        cm_len = len(repo.commits[cm_idx].body) if cm_idx != -1 else 0
        is_len = len(repo.issues[is_idx].body)  if is_idx != -1 else 0

        total = cm_len + is_len + rm_len or 1

        w_rm = rm_len / total
        w_cm = cm_len / total
        w_is = is_len / total

        return w_rm * readme_score + w_cm * cm_score + w_is * is_score



    
    def rank(self, query:  str, top_k = 25):
        keywords = preprocess_query(query)
        print(f"\nQuery: {query}")
        print(f"Keywords: {keywords}")


        readme_texts = [repo.readme for repo in self.repositories]

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(readme_texts)
        query_vector = vectorizer.transform([" ".join(keywords)])
        readme_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

        top_results = list(enumerate(readme_scores))

        ranked = sorted(top_results, key=lambda x: x[1], reverse=True)[:top_k]

        result = []

        for idx, rm_score in ranked:
            repo = self.repositories[idx]
            max_cm_score = self.max_commit_score(query_vector, vectorizer, repo)
            max_is_score = self.max_issue_score(query_vector, vectorizer, repo)
            w_score = self.weighted_score(repo, rm_score, max_cm_score,  max_is_score)
            result.append((idx, w_score))

        return result




# def test_ranker():
#     with open("db/sample_data.json") as f:
#         json_data = json.load(f)

#     queries = [
#         "Build a webpage in JavaScript",
#         "Implement caching in Python",
#         "Data analysis using Python",
#         "Optimize loops in C++",
#         "Create a REST API in PHP",
#     ]

#     ranker = Ranker(json_data)

#     for q in queries:
#         ranker.rank(q)
#         print("—" * 60)


# if __name__ == "__main__":
#     test_ranker()










# def rank_repositories(keywords, df):
#     """
#     Use TF-IDF + cosine similarity to rank repos by relevance to keyword query.
#     Returns ranked DataFrame.
#     """
#     if not keywords:
#         return pd.DataFrame()
    
#     query_text = " ".join(keywords)
#     vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

#     tfidf_matrix = vectorizer.fit_transform(df["readme"])
#     query_vector = vectorizer.transform([query_text])

#     similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     df["similarity"] = similarity_scores

#     ranked = df.sort_values(by="similarity", ascending=False)
#     return ranked[["repo_name","readme_raw", "language", "similarity"]]


# def process_query_and_rank(json_file : dict, query : str):
#     """
#     Given a user query, extract keywords and rank matching repos from dataset.
#     Returns ranked result or empty list.
#     """
#     keywords = preprocess_query(query)
#     print(f"\nQuery: {query}")
#     print(f"Keywords: {keywords}")
#     print(f"Language: {list(json_file.keys())}")

#     df = parse_data(json_file)

#     # if language:
#     #     df = df[df["language"] == language.lower()]
#     #     if df.empty:
#     #         print(f"No repositories found for language: {language}")
#     #         return pd.DataFrame()

#     ranked = rank_repositories(keywords, df)

#     if ranked.empty or ranked["similarity"].max() == 0:
#         print("No matching repositories found.")
#     else:
#         print("Top Results:")
#         print(ranked.head())

#     return ranked


# def test_ranker():
#     json_path = "db/sample_data.json"
#     queries = [
#         "Build a webpage in JavaScript",
#         "Implement caching in Python",
#         "Data analysis using Python",
#         "Optimize loops in C++",
#         "Create a REST API in PHP",
#     ]

#     for q in queries:
#         process_query_and_rank(json_path, q)
#         print("—" * 60)


# if __name__ == "__main__":
#     test_ranker()