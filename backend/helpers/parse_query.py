import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download("punkt_tab")
nltk.download('punkt')
nltk.download('wordnet')

top_langs = [
    "javascript",
    "python",
    "java",
    "typescript",
    "csharp",
    "cpp",
    "php",
    "shell",
    "c",
    "ruby",
]


# def preprocess_query(query, top=True):

#     tokens = nltk.word_tokenize(query.lower())

#     expanded_tokens = set(tokens)
#     for token in tokens:
#         for syn in wordnet.synsets(token):
#             for lemma in syn.lemmas():
#                 expanded_tokens.add(lemma.name().replace("_", " "))

#     return list(expanded_tokens)
def preprocess_query(query: str, top: bool = True, max_synonyms: int = 3) -> list[str]:
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(query.lower())
    filtered_tokens = [
        lemmatizer.lemmatize(t) for t in tokens if t.isalpha()
    ]

    expanded_tokens = set(filtered_tokens)

    for token in filtered_tokens:
        synonyms = set()
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name == token:
                    continue
                if len(name.split()) > 2:
                    continue
                synonyms.add(name)
                if top and len(synonyms) >= max_synonyms:
                    break
            if top and len(synonyms) >= max_synonyms:
                break
        expanded_tokens.update(synonyms)

    return expanded_tokens


def all_langs():

    return [
        "python",
        "java",
        "javascript",
        "c++",
        "c",
        "c#",
        "typescript",
        "ruby",
        "go",
        "rust",
        "swift",
        "kotlin",
        "php",
        "html",
        "css",
        "dart",
        "r",
        "julia",
        "matlab",
        "assembly",
        "fortran",
        "ada",
        "haskell",
        "clojure",
        "ocaml",
        "shell",
        "bash",
        "powershell",
        "perl",
        "sql",
        "verilog",
        "vhdl",
    ]


def test_queries():
    """Test suite for local testing (not run when imported as module)."""
    queries = [
        "Implement cache using Python",
        "Optimize code in C++",
        "Hash map - Java?",
        "Quickest array sorting using JavaScript?",
        "How to efficiently SQL",
    ]

    for query in queries:
        keywords, lang = preprocess_query(query)
        print(f"Query: {query}")
        print(f"Extracted Keywords: {keywords}")
        print(f"Programming Language: {lang}")
        print("_________________________________________________________________")


if __name__ == "__main__":
    test_queries()