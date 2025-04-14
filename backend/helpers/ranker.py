import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from backend.helpers.parse_query import preprocess_query
from .parse_query import preprocess_query


# def load_nested_repositories(json_file):
#     data = json_file
#     print("Sample loaded JSON keys:", list(data.keys())[:3])

#     repo_list = []
#     for language, repos in data.items():
#         for repo_name, repo_info in repos.items():
#             if isinstance(repo_info, dict):
#                 readme_content = repo_info.get("readme", "")
#                 commits = repo_info.get("commits", [])

#                 if isinstance(readme_content, str):
#                     readme_clean = (
#                         readme_content.encode()
#                         .decode("unicode_escape")
#                         .replace("\\n", " ")
#                         .lower()
#                     )

#                     # combine commit headers and bodies
#                     commit_texts = [
#                         f"{c.get('header', '')} {c.get('body', '')}"
#                         for c in commits
#                         if isinstance(c, dict)
#                     ]
#                     commit_content = " ".join(commit_texts).lower()

#                     full_text = f"{readme_clean} {commit_content}"

#                     repo_list.append(
#                         {
#                             "repo_name": repo_name,
#                             "language": language.lower(),
#                             "readme_raw": readme_content,
#                             "full_text": full_text,
#                         }
#                     )


#     return pd.DataFrame(repo_list)import json


def load_repositories_from_flat_json(json_data, min_stars=5, min_forks=3):

    repo_list = []
    for repo_name, repo_info in json_data.items():
        stars = repo_info.get("stars count", 0)
        forks = repo_info.get("forks count", 0)

        if stars < min_stars or forks < min_forks:
            continue

        readme = repo_info.get("readme", "")
        readme_clean = (
            readme.encode().decode("unicode_escape").replace("\\n", " ").lower()
            if isinstance(readme, str)
            else ""
        )

        issues_text = " ".join(
            issue.get("title", "") + " " + issue.get("body", "")
            for issue in repo_info.get("issues", [])
            if isinstance(issue, dict)
        ).lower()

        pr_diffs = " ".join(
            pr.get("diff", "")
            for issue in repo_info.get("issues", [])
            if isinstance(issue, dict)
            for pr in issue.get("pull requests", [])
            if isinstance(pr, dict)
        ).lower()

        combined_text = readme_clean + " " + issues_text + " " + pr_diffs

        repo_list.append(
            {
                "repo_name": repo_name,
                "readme_raw": readme,
                "stars": stars,
                "forks": forks,
                "combined_text": combined_text,
            }
        )

    return pd.DataFrame(repo_list)


def rank_repositories(keywords, df):
    if not keywords:
        return pd.DataFrame()

    query_text = " ".join(keywords[0])
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    query_vector = vectorizer.transform([query_text])
    df["similarity"] = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df["language"] =  keywords[1]

    ranked = df.sort_values(by="similarity", ascending=False)
    return ranked[["repo_name", "readme_raw", "stars", "forks", "similarity"]]


def process_query_and_rank(json_file: dict, query: str):
    keywords = preprocess_query(query)
    print(f"\nQuery: {query}")
    print(f"Keywords: {keywords}")
    print(f"Repositories in Dataset: {len(json_file)}")

    df = load_repositories_from_flat_json(json_file)

    ranked = rank_repositories(keywords, df)

    if ranked.empty or ranked["similarity"].max() == 0:
        print("No matching repositories found.")
    else:
        print("Top Results:")
        print(ranked.head())

    return ranked


def test_ranker():
    import json

    with open("dist_pysample_data.json") as f:
        json_data = json.load(f)

    queries = [
        "Build a web app using Flask in Python",
        "Implement an in-memory cache with Python decorators",
        "Perform data analysis with Pandas in Python",
        "Optimize nested loops using NumPy or list comprehensions in Python",
        "Use Tensorflow or OpenCV for an object detection task",
    ]

    for q in queries:
        process_query_and_rank(json_data, q)
        print("—" * 60)


if __name__ == "__main__":
    test_ranker()


