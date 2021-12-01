import numpy as np # linear algebra
import pandas as pd
from scipy.sparse.construct import rand # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_range_finder
import streamlit as st

import streamlit.components.v1 as components
HtmlFile = open("index.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, width=1200, height=500)

df = pd.read_csv("data/scifi_with_cover.csv")

def combine_features(record):
    try:
        genres = record.genres
        author_name = "_".join(record.author_name.lower().split(" "))
        description = record.book_description
        feature = f"{author_name} {genres} {description}"
        return feature
    except:
        print(temp)

df["combined_features"] = df.apply(combine_features, axis=1)

book_user_like = "Neuromancer"
book_ID = df[df.book_title == book_user_like].id[0]
book_record = df.loc[book_ID]
book_genres = set(book_record.genres.split(" "))

from scipy.sparse import csr_matrix, hstack

def combine_name(author_name):
    return author_name.lower().replace(" ", "_")


uniq_genres_authors = set()
feature_dict = {}

for genre_list in df.genres:
    for genre in genre_list.split(" "):
        uniq_genres_authors.add(genre)
        
for author in df.author_name:
    temp= combine_name(author)
    uniq_genres_authors.add(temp)
        

feature_dict = {item:index for index, item in enumerate(uniq_genres_authors)}

# for index, item in enumerate(uniq_genres):
#     feature_dict[item] = index
    
weight_matrix = np.zeros((df.shape[0], len(feature_dict)))
weight = st.number_input('Enter a number')
st.write(weight)

for index in range(df.shape[0]):
    temp = []
    record = df.loc[index]
    temp.append(combine_name(record.author_name))
    for item in record.genres.split(" "):
        temp.append(item)
        
    for item in temp:
        weight_matrix[index, feature_dict[item]] = weight # Arbitrary weight and can be adjusted

cv = CountVectorizer(stop_words="english")
desc_count_matrix= cv.fit_transform(df["book_description"])

weight_matrix = csr_matrix(weight_matrix)
combined_matrix = hstack((weight_matrix, desc_count_matrix))

cosine_similarity_matrix = cosine_similarity(combined_matrix)

similar_books = list(enumerate(cosine_similarity_matrix[book_ID]))
sorted_similar_books = sorted(similar_books,key=lambda x:x[1],reverse=True)

limit = 20
print(book_record.book_title, book_record.author_name, book_record.genres)
for element in sorted_similar_books[1:limit]:
    book_index = element[0]
    recommendation = df[df.id == book_index]["book_title"].values[0] 
    author = df[df.id == book_index]["author_name"].values[0]
    genres = df[df.id == book_index]["genres"].values[0]
    genres = set(genres.split(" "))
    common_genres = book_genres.intersection(genres)
    st.write(f"Book: {recommendation.upper()} Author:{author.upper()} and Genres: {common_genres}")

random_books = np.random.randint(0, df.shape[0], 10)
random_books_cover = df.iloc[random_books].cover.values

# for cover in random_books_cover:
#     st.image(cover)