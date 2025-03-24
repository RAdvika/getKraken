import re
import nltk
from nltk.corpus import wordnet

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


def preprocess_query(query, top=True):

    tokens = nltk.word_tokenize(query.lower())

    expanded_tokens = set(tokens)
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded_tokens.add(lemma.name().replace("_", " "))

    return list(expanded_tokens)


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

