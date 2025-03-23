import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.helpers.parse_query import preprocess_query


def load_nested_repositories(json_file):
    """
    Load nested dataset structured as dataset[language][repo] from JSON file.
    Returns a flat dataframe with columns: ['repo_name', 'language', 'readme']
    """

    with open(json_file, "r") as file:
        data = json.load(file)
    print("Sample loaded JSON keys:", list(data.keys())[:3])

    repo_list = []
    for language, repos in data.items():
        for repo_name, repo_info in repos.items():
            if isinstance(repo_info, dict):
                readme_content = repo_info.get("readme", "")
                if isinstance(readme_content, str):
                    readme_clean = (
                        readme_content.encode()
                        .decode("unicode_escape")
                        .replace("\\n", " ")
                        .lower()
                    )
                    repo_list.append(
                        {
                            "repo_name": repo_name,
                            "language": language.lower(),
                            "readme": readme_clean,
                        }
                    )

    return pd.DataFrame(repo_list)


def rank_repositories(keywords, df):
    """
    Use TF-IDF + cosine similarity to rank repos by relevance to keyword query.
    Returns ranked DataFrame.
    """
    if not keywords:
        return pd.DataFrame()

    query_text = " ".join(keywords)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(df["readme"])
    query_vector = vectorizer.transform([query_text])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df["similarity"] = similarity_scores

    ranked = df.sort_values(by="similarity", ascending=False)
    return ranked[["repo_name", "language", "similarity"]]


def process_query_and_rank(json_file, query):
    """
    Given a user query, extract keywords and rank matching repos from dataset.
    Returns ranked result or empty list.
    """
    keywords, language = preprocess_query(query)
    print(f"\nQuery: {query}")
    print(f"Keywords: {keywords}")
    print(f"Language: {language}")

    df = load_nested_repositories(json_file)

    if language:
        df = df[df["language"] == language.lower()]
        if df.empty:
            print(f"No repositories found for language: {language}")
            return pd.DataFrame()

    ranked = rank_repositories(keywords, df)

    if ranked.empty or ranked["similarity"].max() == 0:
        print("No matching repositories found.")
    else:
        print("Top Results:")
        print(ranked.head())

    return ranked


def test_ranker():
    json_path = "db/sample_data.json"
    queries = [
        "Build a webpage in JavaScript",
        "Implement caching in Python",
        "Data analysis using Python",
        "Optimize loops in C++",
        "Create a REST API in PHP",
    ]

    for q in queries:
        process_query_and_rank(json_path, q)
        print("—" * 60)


if __name__ == "__main__":
    test_ranker()

