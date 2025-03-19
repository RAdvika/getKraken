#incomplete rankings 

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parse_query import preprocess_query 

def load_repositories(json_file):
    """ Load repositories from JSON and clean data. """
    with open(json_file, "r") as file:
        data = json.load(file)

    repo_list = []
    for repo_name, readme_content in data.items():
        readme_clean = readme_content.encode().decode("unicode_escape").replace("\\n", " ")
        repo_list.append({"repo_name": repo_name, "readme": readme_clean})

    return pd.DataFrame(repo_list)

def rank_repositories(query, df):
    """ Rank repositories using TF-IDF and cosine similarity. """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["readme"])
    if isinstance(query, list):
        query = " ".join(query)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    df["similarity"] = similarity_scores
    ranked_repos = df.sort_values(by="similarity", ascending=False)
    return ranked_repos[["repo_name", "similarity"]]

# if __name__ == "__main__":
#     df = load_repositories("backend/sample_data.json")
    
#     query = "How to implement LRU cache in Python?"
#     keywords, lang = preprocess_query(query)
#     ranked_results = rank_repositories(keywords, df) 
    
#     print(ranked_results.head())

