import re
import nltk
from nltk.corpus import wordnet

def preprocess_query(query):
    programming_languages = [
    "python", "java", "javascript", "c++", "c", "c#", "typescript", "ruby", "go", "rust", "swift", "kotlin", 
    "php", "html", "css", "typescript", "dart",

    "r", "julia", "matlab",
    "assembly", "fortran", "ada",
    "haskell", "clojure", "elixir", "erlang", "f#", "ocaml",

    "shell", "bash", "powershell", "perl",
    "sql",
    "verilog", "vhdl"] 
    
    #TODO:include top 10 for later

    tokens = nltk.word_tokenize(query.lower())

    language = None
    for token in tokens:
        if token in programming_languages:
            language = token
            tokens.remove(token)

    expanded_tokens = set(tokens)
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded_tokens.add(lemma.name().replace("_", " "))

    return list(expanded_tokens), language

queries = [
    "Implement cache using Python",
    "Optimize code in C++",
    "Hash map - Java?",
    "Quickest array sorting using JavaScript?",
    "How to efficiently SQL"
]

for query in queries: 
    keywords, lang = preprocess_query(query)
    print(f"Query: {query}")
    print(f"Extracted Keywords: {keywords}")
    print(f"Programming Language: {lang}")
    print("_________________________________________________________________")
