import time
import numpy as np  # linear algebra
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit.components.v1 as components

start_time = time.perf_counter()

# Loading data
df = pd.read_csv("data/scifi_with_cover.csv")
ROWS, COLUMNS = df.shape

# LAYOUTS
st.set_page_config(page_title="Sci-fi Rec System", layout="wide")

# Header Layout
header_text = """# Sci-fi Book Recommendation System
> *Choose your **favorite book** and find the one similar to your preference with manually adjustable weights and genres.*
"""
st.markdown(header_text)

# Sidebar Layout
st.sidebar.title("Preferences")
book_user_like = st.sidebar.text_input("Enter the book you like", value="Neuromancer")
author_weight = st.sidebar.number_input("Author weight", value=50)
genre_weight = st.sidebar.number_input("Genre weight", value=50)
preferred_genre = st.sidebar.text_input("Enter a genre", value="cyberpunk")
st.sidebar.write(book_user_like)

try:
    book_user_like = book_user_like.strip().lower()
    book_ID = df[df.book_title.apply(lambda x: x.strip().lower()) == book_user_like].id.values[0]
except KeyError:
    st.warning("Sorry! No such book is found")
    st.stop()

book_record = df.loc[book_ID]
book_genres = set(book_record.genres.split(" "))

book_cover = book_record.cover[0]


uniq_genres = set()
uniq_authors = set()
feature_dict = {}

for genre_list in df.genres:
    for genre in genre_list.split(" "):
        uniq_genres.add(genre)

for author in df.author_name:
    author = author.strip().lower().replace(" ", "_")
    uniq_authors.add(author)

uniq_genres_authors = uniq_authors.union(uniq_genres)

author_feature_dict = {item: index for index, item in enumerate(uniq_authors)}
genre_feature_dict = {item: index for index, item in enumerate(uniq_genres)}


@st.cache
def get_author_weight_matrix(df, author_feature_dict):
    author_weight_matrix = np.zeros((df.shape[0], len(author_feature_dict)))

    for index in range(ROWS):
        record = df.loc[index]
        author = record.author_name.strip().lower().replace(" ", "_")
        author_weight_matrix[index, author_feature_dict[author]] = author_weight  

    return csr_matrix(author_weight_matrix)


    
@st.cache
def get_genre_weight_matrix(df, genre_feature_dict):
    genre_weight_matrix = np.zeros((df.shape[0], len(genre_feature_dict)))

    for index in range(ROWS):
        temp = []
        record = df.loc[index]
        for item in record.genres.split(" "):
            temp.append(item)

        for item in temp:
            genre_weight_matrix[index, genre_feature_dict[item]] = genre_weight 

    return csr_matrix(genre_weight_matrix)


@st.cache
def get_desc_count_matrix(df):
    cv = CountVectorizer(stop_words="english")
    desc_count_matrix = cv.fit_transform(df["book_description"])
    return desc_count_matrix

combined_matrix = get_desc_count_matrix(df)

if not int(author_weight) == 0:
    author_weight_matrix =  get_author_weight_matrix(df, author_feature_dict)
    combined_matrix = hstack((combined_matrix, author_weight_matrix))

if not int(genre_weight) == 0:
    genre_weight_matrix =  get_genre_weight_matrix(df, genre_feature_dict)
    combined_matrix = hstack((combined_matrix, genre_weight_matrix))


@st.cache
def get_cosine_similarity(combined_matrix):
    return cosine_similarity(combined_matrix)

cosine_similarity_matrix = get_cosine_similarity(combined_matrix)

similar_books = list(enumerate(cosine_similarity_matrix[book_ID]))
sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

limit = 6
st.caption(
    f" Current book: {book_record.book_title}, {book_record.author_name}, {book_record.genres}"
)

recommended_books = [item[0] for item in sorted_similar_books[1:limit]]
df_rec = df.loc[
    recommended_books,
    ["cover", "book_title", "author_name", "book_description", "rating_score"],
]

# st.write(df.loc[recommended_books, ["cover", "book_title", "author_name", "book_description", "rating_score"]])

with open("layouts/results.md") as f:
    page = f.read()
    for i in range(limit - 1):
        temp = df_rec.iloc[i, :].values
        str_ = f'| <img src="{temp[0]}" width="150"> | **{temp[1]}** | {temp[2]} | {temp[3][:500]} | {temp[4]} |'
        page = page + "\n" + str_

    st.markdown(page, unsafe_allow_html=True)

# for element in sorted_similar_books[1:limit]:
#     book_index = element[0]
#     recommendation = df[df.id == book_index]["book_title"].values[0]
#     author = df[df.id == book_index]["author_name"].values[0]
#     genres = df[df.id == book_index]["genres"].values[0]
#     genres = set(genres.split(" "))
#     common_genres = book_genres.intersection(genres)
#     cover = df[df.id == book_index]["cover"].values[0]
#     st.write(recommendation)
#     st.write(f"{author.upper()} ")
#     st.write(f"{common_genres}")
#     st.image(cover, width=200)



random_books = np.random.randint(0, df.shape[0], 10)
random_books_cover = df.iloc[random_books].cover.values

finish_time = time.perf_counter()
passed_time = finish_time - start_time
st.caption(f"Recommendation is calculated in {round(passed_time,3)}s")
