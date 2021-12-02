import time
import numpy as np  # linear algebra
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import streamlit.components.v1 as components

start_time = time.perf_counter()

st.set_page_config(layout="wide")


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


def combine_name(author_name):
    return author_name.lower().replace(" ", "_")

@st.cache
def process_data(df):
    df["combined_features"] = df.apply(combine_features, axis=1)
    return df

df = process_data(df)



# LAYOUTS

# Header Layout
header_text ="""# Sci-fi Book Recommendation System
Choose your **favorite book** and find the one similar to your preference with manually adjustable weights and genres.
"""
st.markdown(header_text)

# Sidebar Layout
st.sidebar.title("Preferences")
book_user_like = st.sidebar.text_input("Enter the book you like", value="Neuromancer")
weight = st.sidebar.number_input("Enter a weight", value=50)
preferred_genre = st.sidebar.text_input("Enter a genre")

# Default Values
# book_user_like = "Neuromancer"
# weight = 100

try:
    book_ID = df[df.book_title == book_user_like].id.values[0]
    book_record = df.loc[book_ID]
    book_genres = set(book_record.genres.split(" "))
    book_cover = df[df.book_title == book_user_like].cover.values[0]


    uniq_genres_authors = set()
    feature_dict = {}

    for genre_list in df.genres:
        for genre in genre_list.split(" "):
            uniq_genres_authors.add(genre)

    for author in df.author_name:
        temp = combine_name(author)
        uniq_genres_authors.add(temp)


    feature_dict = {item: index for index, item in enumerate(uniq_genres_authors)}



    weight_matrix = np.zeros((df.shape[0], len(feature_dict)))

    for index in range(df.shape[0]):
        temp = []
        record = df.loc[index]
        temp.append(combine_name(record.author_name))
        for item in record.genres.split(" "):
            temp.append(item)

        for item in temp:
            weight_matrix[
                index, feature_dict[item]
            ] = weight  # Arbitrary weight and can be adjusted

    cv = CountVectorizer(stop_words="english")
    desc_count_matrix = cv.fit_transform(df["book_description"])

    weight_matrix = csr_matrix(weight_matrix)
    combined_matrix = hstack((weight_matrix, desc_count_matrix))

    cosine_similarity_matrix = cosine_similarity(combined_matrix)

    similar_books = list(enumerate(cosine_similarity_matrix[book_ID]))
    sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

    limit = 6
    st.caption(f" Current book: {book_record.book_title}, {book_record.author_name}, {book_record.genres}")

    recommended_books = [item[0] for item in sorted_similar_books[1:limit]]
    df_rec = df.loc[recommended_books, ["cover", "book_title", "author_name", "book_description","rating_score"]]

    # st.write(df.loc[recommended_books, ["cover", "book_title", "author_name", "book_description", "rating_score"]])

    with open("layouts/results.md") as f:
        page = f.read()
        for i in range(limit-1):
            temp = df_rec.iloc[i,:].values
            str_ = f'| <img src="{temp[0]}" width="150"> | **{temp[1]}** | {temp[2]} | {temp[3][:500]} | {temp[4]} |'
            page = page + "\n" + str_

        st.markdown(page,unsafe_allow_html=True)

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
except:
    st.error("Sorry! No such book is found")

    random_books = np.random.randint(0, df.shape[0], 10)
    random_books_cover = df.iloc[random_books].cover.values

finish_time = time.perf_counter()
passed_time = finish_time -start_time
st.caption(f"Recommendation is calculated in {round(passed_time,3)}s")