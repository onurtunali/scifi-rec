import time
import numpy as np  # linear algebra
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit.components.v1 as components
start_time = time.perf_counter()

st.set_page_config(page_title="Sci-fi Rec System", layout="wide")

# Loading data
@st.cache
def load_data(data_path):
    return pd.read_csv(data_path)

data_path = "data/scifi_with_cover.csv"
df_org = load_data(data_path)
ROWS, COLUMNS = df_org.shape

uniq_genres = set()
feature_dict = {}

for genre_list in df_org.genres:
    for genre in genre_list.split(" "):
        uniq_genres.add(genre.capitalize().replace("_", " "))
genre_list = sorted(list(uniq_genres))
# LAYOUTS

# Header Layout
st.markdown("# Sci-fi Book Recommendation System")

# Sidebar Layout
st.sidebar.title("Preferences")

query_params = st.experimental_get_query_params()
print(query_params)

if query_params:
    book_user_like = st.sidebar.text_input("Enter the book you like", value=query_params['book_title'][0])
else:
    book_user_like = st.sidebar.text_input("Enter the book you like", value="Neuromancer")


author_weight = st.sidebar.number_input("Author weight", value=0)
genre_weight = st.sidebar.number_input("Genre weight", value=0)
preferred_genre = st.sidebar.selectbox("Choose a genre", genre_list, index=135)
st.sidebar.markdown("*Choose your **favorite book** and find the one similar to your preference with manually adjustable weights and genres.*")
st.sidebar.info(f"**Current book:** {book_user_like}")

try:
    book_user_like = book_user_like.strip().lower()
    book_ID = df_org[df_org.book_title.apply(lambda x: x.strip().lower()) == book_user_like].id.values[0]
except:
    st.warning("Sorry! No such book is found")
    st.stop()


genre_index = [preferred_genre.lower().replace(" ", "_") in item for item in df_org.genres]
genre_index[book_ID] = True

df = df_org.loc[genre_index].reset_index(drop=True)
ROWS, COLUMNS = df.shape
if ROWS < 3:
    st.warning("There is not enough records for this genre :(")
    st.stop()

book_index = df[df.id == book_ID].index.values[0]

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


def get_author_weight_matrix(df, author_feature_dict):
    author_weight_matrix = np.zeros((df.shape[0], len(author_feature_dict)))

    for index in range(ROWS):
        record = df.iloc[index]
        author = record.author_name.strip().lower().replace(" ", "_")
        author_weight_matrix[index, author_feature_dict[author]] = author_weight  

    return csr_matrix(author_weight_matrix)


def get_genre_weight_matrix(df, genre_feature_dict):
    genre_weight_matrix = np.zeros((df.shape[0], len(genre_feature_dict)))

    for index in range(ROWS):
        temp = []
        record = df.iloc[index]
        for item in record.genres.split(" "):
            temp.append(item)

        for item in temp:
            genre_weight_matrix[index, genre_feature_dict[item]] = genre_weight 

    return csr_matrix(genre_weight_matrix)

def get_desc_count_matrix(df):
    cv = CountVectorizer(stop_words="english", max_features=5000)
    desc_count_matrix = cv.fit_transform(df["book_description"])
    return desc_count_matrix

desc_count_matrix = get_desc_count_matrix(df)
is_author_weight = False
is_genre_weight = False

if not int(author_weight) == 0:
    is_author_weight = True
    author_weight_matrix =  get_author_weight_matrix(df, author_feature_dict)
    

if not int(genre_weight) == 0:
    is_genre_weight = True
    genre_weight_matrix =  get_genre_weight_matrix(df, genre_feature_dict)

def get_cosine_similarity(combined_matrix):
    return cosine_similarity(combined_matrix)

if is_author_weight and is_genre_weight:
    combined_matrix = hstack((author_weight_matrix, genre_weight_matrix, desc_count_matrix))

if not is_author_weight and not is_genre_weight :
    combined_matrix = desc_count_matrix.copy()

if is_author_weight and not is_genre_weight:
    combined_matrix = hstack((author_weight_matrix, desc_count_matrix))

if not is_author_weight and is_genre_weight:
    combined_matrix = hstack((genre_weight_matrix, desc_count_matrix))

cosine_similarity_matrix = get_cosine_similarity(combined_matrix)

similar_books = list(enumerate(cosine_similarity_matrix[book_index]))
sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

limit = 5

recommended_books = [item[0] for item in sorted_similar_books[1:limit]]

with open("layouts/results.md") as f:
    page = f.read()
    for index in recommended_books:
        record = df.iloc[index]
        cover = record.cover.replace("\n", "")
        title = record.book_title.replace("\n", "")
        author = record.author_name.replace("\n", "")
        description = record.book_description.replace("\n", "")
        rating = record.rating_score
        str_ = f'| <img src="{cover}" width="100"> | **{title}** | {author} | {description[:500]} | {rating} |'
        str_ = str_.replace("\n", "")
        page = page + "\n" + str_

    st.markdown(page, unsafe_allow_html=True)

finish_time = time.perf_counter()
passed_time = finish_time - start_time
st.caption(f"Recommendation is calculated in {round(passed_time,3)}s")
