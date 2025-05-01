from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS  
from sklearn.metrics.pairwise import cosine_similarity
from .parse_query import preprocess_query
from typing import List, Callable, Tuple, Any
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from collections import defaultdict
import re

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
custom_stopwords = {
            'add',
            'adds',
            'added',
            'fix',
            'bugfix',
            'bugfixes',
            'fixes',
            'update',
            'test',
            'tests',
            'remove',
            'line',
            'issue',
            'help',
            'using',
            'change',
            'changes',
            'changed',
            'pr',
            'does',
            'support',
            'merge',
            'pip',
            'error',
            'errors',
            'sudo',
            'install',
            'app',
            'apps',
            'python',
            'use',
            'fixed',
            'described',
            'describe'
        }
custom_stopwords = custom_stopwords| ENGLISH_STOP_WORDS

class Issue:
    def __init__(self, title : str, body: str, url: str):
        self.title = title
        self.body = body
        self.url = url


class Commit:
    def __init__(self, sha : str, header: str, body:  str,  url: str, author: str):
        self.sha = sha
        self.header = header
        self.body = body
        self.url = url
        self.author = author

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


class Result:
    def __init__(self, repo_idx : int,
                 commit_idx : int,
                 issue_indx: int,
                 features: list[str],
                 sim_score: float,
                 hidden: list[str]
    ):
        self.repo_idx = repo_idx
        self.commit_idx = commit_idx
        self.issue_idx = issue_indx
        self.coocmatrix_feature = features
        self.sim_score = sim_score
        self.hidden_d = hidden

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


        self.readme_pipeline = make_pipeline(
            TfidfVectorizer(max_features=10000, stop_words=list(custom_stopwords),ngram_range=(1, 3)),
            TruncatedSVD(n_components=800),
            Normalizer(copy=False)
        )
        self.commit_pipeline = make_pipeline(
            TfidfVectorizer(max_features=30, stop_words=list(custom_stopwords))
            )
        
        self.issue_pipeline = make_pipeline(
            TfidfVectorizer(max_features=200, stop_words=list(custom_stopwords),ngram_range=(1, 2)),
            TruncatedSVD(n_components=10),
        )

        self.readme_matrix = self.readme_pipeline.fit_transform(readme_texts)
        self.commit_lsa = self.commit_pipeline.fit(commit_texts)
        self.issues_lsa = self.issue_pipeline.fit(issue_texts)



        self.commit_coom =[
            self.build_co_occurrence_matrix(
                [commit.body + " " + commit.header for commit in repo.commits]
            )
            for repo in self.repositories
        ]


        del readme_texts
        del commit_texts
        del issue_texts


    def clean_text(self, texts: List[str], max_len: int = 15) -> List[str]:
        cleaned_texts = []
        pattern = re.compile(r'^[a-z]{1,%d}$' % max_len)

        for text in texts:
            lower_text = text.lower()
            tokens = lower_text.split()

            filtered_tokens: List[str] = []
            for token in tokens:
                if not pattern.match(token):
                    continue
                if token.isdigit():
                    continue
                if token in custom_stopwords:
                    continue
                filtered_tokens.append(token)

            cleaned_string = " ".join(filtered_tokens)
            cleaned_texts.append(cleaned_string)

        return cleaned_texts




    def parse_data(self, json_data) -> list[Repository]:
        repositories = []
        for repo_name, repo_data in json_data.items():
            if repo_name in black_list:
                continue
            issues = [Issue(issue['title'], issue['body'], issue['url']) for issue in repo_data.get('issues', [])]
            commits = [Commit(commit['sha'], commit['header'], commit['body'], commit['url'], commit['author']) for commit in repo_data.get('commits', [])]

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
    
    def get_top_svd_terms(
        self,
        text: str,
        vectorizer_pipeline,
        top_k: int = 5
    ) -> List[str]:
        tfidf: TfidfVectorizer = vectorizer_pipeline.named_steps['tfidfvectorizer']
        svd: TruncatedSVD = vectorizer_pipeline.named_steps.get('truncatedsvd')
        tf = tfidf.transform([text])              
        latent = svd.transform(tf)                
        approx = latent.dot(svd.components_)  
        top_idxs = np.argsort(approx[0])[::-1][:top_k]
        names = tfidf.get_feature_names_out()

        return [names[i] for i in top_idxs]

    def get_top_hidden_dims(
        self,
        query_vector: np.ndarray,
        readme_idx: int,
        top_k: int = 5) -> List[str]:
        min_contrib = 0.01

        #dimension-wise contributions
        qv = query_vector.flatten()
        dv = self.readme_matrix[readme_idx]
        contrib = qv * dv

        valid_dims = np.where(contrib >= min_contrib)[0]
        if valid_dims.size == 0:
            return []


        sorted_dims = valid_dims[np.argsort(contrib[valid_dims])[::-1]]
        top_dims = sorted_dims[:top_k]

        # SVD and TF-IDF to map dims to words
        tfidf: TfidfVectorizer = self.readme_pipeline.named_steps['tfidfvectorizer']
        svd: TruncatedSVD = self.readme_pipeline.named_steps['truncatedsvd']
        feature_names = tfidf.get_feature_names_out()

        #top words across all top dims
        flat_words = []
        for dim in top_dims:
            comp = svd.components_[dim]
            top_idxs = np.argsort(comp)[::-1][:1]
            flat_words.extend(feature_names[i] for i in top_idxs)

        # this needs to be in loop to maintain order
        seen = set()
        flat_unique = [w for w in flat_words if not (w in seen or seen.add(w))]
        return flat_unique


    def max_similarity_score(self, 
        query_vector, 
        vectorizer_pipeline, 
        items: list[Any], 
        text_extractor: Callable[[Any], str]):
        if not items:
            return -1, 0.0

        item_texts = [text_extractor(item) for item in items]
        item_texts = self.clean_text(item_texts)
        item_vectors = vectorizer_pipeline.transform(item_texts)
        similarities = cosine_similarity(query_vector, item_vectors).flatten()

        max_index = int(np.argmax(similarities))
        max_score = similarities[max_index]

        if max_score <= 0.01:
            return  -1, 0.0

        return (max_index, max_score)



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

    def build_co_occurrence_matrix(self, corpus: List[str]):
        cleaned_corpus = self.clean_text(corpus)
        vectorizer = CountVectorizer(stop_words=list(custom_stopwords), max_features=12500, ngram_range=(1, 1))
        word_count_matrix = vectorizer.fit_transform(cleaned_corpus)
        
        # Create the co-occurrence matrix (using the dot product of word count matrix)
        co_occurrence_matrix = (word_count_matrix.T @ word_count_matrix).tocoo()
        return co_occurrence_matrix, vectorizer.get_feature_names_out()
    

    def expand_query(self, query: str,coom ,vocab, top_k: int = 10) -> List[str]:
        query_terms = query.split() 
        query_indices = [np.where(vocab == term)[0][0] for term in query_terms if term in vocab]
        co_occurrence_scores = defaultdict(float)
        for idx in query_indices:
            row_indices = coom.row[coom.col == idx]
            co_occurrence_values = coom.data[coom.col == idx]
            
            for row_idx, value in zip(row_indices, co_occurrence_values):
                co_occurrence_scores[vocab[row_idx]] += value

        expanded_terms = sorted(co_occurrence_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # for t,s  in expanded_terms:
        #     print(f"score:{s} term:{t}")
        
        return [term for term, score in expanded_terms]



    
    def rank(self, query: str, top_k=56):
        #preprocess query
        keywords = preprocess_query(query)  
        print(f"\nQuery: {query}")
        print(f"Keywords WordNet: {keywords}")

        #score against READMEs to get your top_k candidates
        query_vec_rm   = self.readme_pipeline.transform([" ".join(keywords)])
        readme_scores  = cosine_similarity(query_vec_rm, self.readme_matrix).flatten()
        top_results    = sorted(enumerate(readme_scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, rm_score in top_results:
            repo = self.repositories[idx]
            hidden = self.get_top_hidden_dims(query_vec_rm, idx)

            coom, vocab = self.commit_coom[idx]

            expanded_terms = self.expand_query(
                " ".join(keywords), 
                coom, vocab,
                top_k=10
            )

            all_terms = set(keywords) | set(expanded_terms)
            expanded_query_str = " ".join(all_terms)

            q_cm = self.commit_pipeline.transform([expanded_query_str])
            q_is = self.issue_pipeline.transform([expanded_query_str])

            cm_idx, cm_score = self.max_similarity_score(
                q_cm, self.commit_lsa, repo.commits, lambda c: c.body + " " + c.header
            )
            is_idx, is_score = self.max_similarity_score(
                q_is, self.issues_lsa, repo.issues, lambda i: i.body + " " + i.title
            )

            w_score = self.weighted_score(repo, rm_score, (cm_idx, cm_score), (is_idx, is_score))
            feats = expanded_terms
            if w_score > 0.05:
                print(f"Repo #{idx} top hidden dims: {hidden}")
                print(f"Repo #{idx} COMMIT: {cm_score} \t ISSUE: {is_score}")
                print(f"Repo #{idx} expanded query: {expanded_terms}\n")
                results.append((idx, cm_idx, is_idx ,feats , w_score, hidden))
            
        rank_result = sorted(results, key=lambda x: x[3], reverse=True)



        return [Result(idx, cm_idx, is_idx ,feats , w_score, hidden) for idx, cm_idx, is_idx ,feats , w_score, hidden in rank_result]
