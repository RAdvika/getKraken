from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from backend.helpers.parse_query import preprocess_query
from .parse_query import preprocess_query
from typing import List, Callable, Tuple, Any
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np


black_list = {
    'TheAlgorithms/Python',
    'nodejs/node',
    'jakevdp/PythonDataScienceHandbook',
    'facebook/buck',
    'microsoft/Data-Science-For-Beginners',
}

"""
REMOVE LIST

TheAlgorithms/Python
nodejs/node
jakevdp/PythonDataScienceHandbook
facebook/buck
microsoft/Data-Science-For-Beginners


"""


class Issue:
    def __init__(self, title : str, body: str, url: str):
        self.title = title
        self.body = body
        self.url = url


class Commit:
    def __init__(self, sha : str, header: str, body:  str,  url: str):
        self.sha = sha
        self.header = header
        self.body = body
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

class Ranker:
    def __init__(self, json_data: dict):
        self.repositories = self.parse_data(json_data)

        readme_texts = [repo.readme for repo in self.repositories]
        readme_texts  =  self.clean_text(readme_texts)

        commit_texts = [
            commit.body + " " + commit.header
            for repo in self.repositories
            for commit in repo.commits
        ]
        commit_texts = self.clean_text(commit_texts)


        issue_texts = [
            issue.body + " " + issue.title
            for repo in self.repositories
            for issue in repo.issues
        ]
        issue_texts = self.clean_text(issue_texts)


        self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words="english",
                    ngram_range=(1, 2)
                )
        
        self.svd        = TruncatedSVD(n_components=10)
        self.normalizer = Normalizer()

        self.readme_pipeline = make_pipeline(
            TfidfVectorizer(max_features=2000, stop_words="english",ngram_range=(1, 3)),
            TruncatedSVD(n_components=200),
            Normalizer(copy=False)
        )
        self.commit_pipeline = make_pipeline(
            TfidfVectorizer(max_features=30, stop_words="english"),
        )
        
        self.issue_pipeline = make_pipeline(
            TfidfVectorizer(max_features=50, stop_words="english",ngram_range=(1, 2)),
            TruncatedSVD(n_components=10),
        )
        # self.readme_matrix = self.lsa_pipeline.fit_transform(readme_texts)
        self.readme_matrix = self.readme_pipeline.fit_transform(readme_texts)
        
        self.commit_lsa = self.commit_pipeline.fit(commit_texts)
        self.issues_lsa = self.issue_pipeline.fit(issue_texts)


    def clean_text(self, texts: List[str], max_len: int = 15) -> List[str]:
        cleaned_texts = []
        for text in texts:
            words = text.split()
            # Filter out words longer than max_len or containing backslash or forward slash
            filtered = [word for word in words if len(word) <= max_len and '\\' not in word and '/' not in word]
            cleaned_texts.append(" ".join(filtered))
        return cleaned_texts


    def parse_data(self, json_data) -> list[Repository]:
        repositories = []
        for repo_name, repo_data in json_data.items():
            if repo_name in black_list:
                continue
            issues = [Issue(issue['title'], issue['body'], issue['url']) for issue in repo_data.get('issues', [])]
            commits = [Commit(commit['sha'], commit['header'], commit['body'], commit['url']) for commit in repo_data.get('commits', [])]

            repo = Repository(
                repo_name=repo_name,
                readme=repo_data.get('readme', ''),
                short_desc=repo_data.get('short desc', ''),
                forks_count=repo_data.get('forks count', -1),
                stars_count=repo_data.get('stars count', -1),
                issues_count=repo_data.get('issues count', -1),
                issues=issues,
                commits=commits
            )
            repositories.append(repo)
        return repositories
    
    def max_similarity_score(self, 
        query_vector, 
        vectorizer_pipeline, 
        items: list[Any], 
        text_extractor: Callable[[Any], str]
    ) -> Tuple[int, float]:
        if not items:
            return -1, 0.0

        item_texts = [text_extractor(item) for item in items]
        item_vectors = vectorizer_pipeline.transform(item_texts)
        similarities = cosine_similarity(query_vector, item_vectors).flatten()

        max_index = int(np.argmax(similarities))
        max_score = similarities[max_index]
        return max_index, max_score



    def weighted_score(self, 
                       repo:  Repository, 
                       readme_score: float,
                       commit_res: Tuple[int, float], 
                       issue_res: Tuple[int, float]):
        cm_idx, cm_score  = commit_res
        is_idx,  is_score  = issue_res


        rm_len = len(repo.readme)
        cm_len = len(repo.commits[cm_idx].body + repo.commits[cm_idx].header) if cm_idx != -1 else 0
        is_len = len(repo.issues[is_idx].body + repo.issues[is_idx].title)  if is_idx != -1 else 0

        total = cm_len + is_len + rm_len or 1

        w_rm = rm_len / total
        w_cm = cm_len / total
        w_is = is_len / total

        final_score = w_rm * readme_score + w_cm * cm_score + w_is * is_score
        final_score = np.max(final_score, 0)

        # print(f'{[format(w_rm, ".4f"), format(w_cm, ".4f"), format(w_is, ".4f")]}')
        # print(f'{[format(readme_score, ".4f"), format(cm_score, ".4f"), format(is_score, ".4f")]}')
        # print(f'SCORE: {final_score}')
        # print(f'-'*40)


        return final_score



    
    def rank(self, query:  str, top_k = 50):
        #expanded query using NLTK wordNet
        keywords = preprocess_query(query)
        print(f"\nQuery: {query}")
        print(f"Keywords: {keywords}")



        query_vec_rm = self.readme_pipeline.transform([" ".join(keywords)])

        readme_scores = cosine_similarity(query_vec_rm, self.readme_matrix).flatten()

        top_results = sorted(
            list(enumerate(readme_scores)), 
            key=lambda x: x[1], 
            reverse=True
            )[:top_k]

        results = []

        for idx, rm_score in top_results:
            repo = self.repositories[idx]
            query_vec_cm = self.commit_pipeline.transform([" ".join(keywords)])
            query_vec_is = self.issue_pipeline.transform([" ".join(keywords)])

            cm_idx, cm_score = self.max_similarity_score(
                query_vec_cm, 
                self.commit_lsa, 
                repo.commits, 
                lambda c: c.body + ' ' + c.header
            )

            is_idx, is_score = self.max_similarity_score(
                query_vec_is, 
                self.issues_lsa, 
                repo.issues, 
                lambda i: i.body + ' ' + i.title
            )

            w_score = self.weighted_score(
                repo,
                rm_score,
                (cm_idx, cm_score),
                (is_idx, is_score)
            )
            results.append((idx, cm_idx, is_idx, w_score))

        ranked_result = sorted(results, key=lambda x: x[3], reverse=True)
        return ranked_result




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