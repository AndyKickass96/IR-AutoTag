from nltk.featstruct import Feature
from numpy.core.fromnumeric import argsort
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import re

fileq = open("query.txt", "r")
filed1 = open("document1.txt", "r")
filed2 = open("document2.txt", "r")
filed3 = open("document3.txt", "r")

q = fileq.read().lower()
d1 = filed1.read().lower()
d2 = filed2.read().lower()
d3 = filed3.read().lower()

# q = q.lower()
# d1 = d1.lower()
# d2 = d2.lower()
# d3 = d3.lower()

# q = "President Trump and Putin".lower()
# d1 = "Mr Trump became president after winning the political election.".lower()
# d2 = "President Trump says Putin had no political interference in the election outcome.".lower()
# d3 = "Post election, Vladimir Putin became President of Russia.".lower()

processed_q = re.sub(r'[^\w\s]', '', q)
processed_d1 = re.sub(r'[^\w\s]', '', d1)
processed_d2 = re.sub(r'[^\w\s]', '', d2)
processed_d3 = re.sub(r'[^\w\s]', '', d3)

documents = [processed_q, processed_d1, processed_d2, processed_d3]

# x = CountVectorizer(stop_words='english')
x = TfidfVectorizer(stop_words='english')
vector = x.fit_transform(documents)
cosim = cosine_similarity(vector[0], vector)
urutan = cosim.argsort()
print(x.vocabulary_)
print("Query: ", processed_q)
print("D1: ", processed_d1)
print("D2: ", processed_d2)
print("D3: ", processed_d3)
print()
print(cosim)
print("PALING MIRIP DENGAN QUERY: D", urutan[0][len(documents)-2])
print("COSIM", cosim[0][urutan[0][len(documents)-2]])
