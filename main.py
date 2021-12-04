import time
import numpy as np  # linear algebra
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from memory_profiler import profile

start_time = time.perf_counter()

# Loading data
df = pd.read_csv("data/scifi_with_cover.csv")
ROWS, COLUMNS = df.shape

book_user_like = "Neuromancer"
author_weight = 50
genre_weight = value=50

try:
    book_user_like = book_user_like.strip().lower()
    book_ID = df[df.book_title.apply(lambda x: x.strip().lower()) == book_user_like].id.values[0]
except:
    print("Sorry! No such book is found")

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


@profile
def get_author_weight_matrix(df, author_feature_dict):
    author_weight_matrix = np.zeros((df.shape[0], len(author_feature_dict)))

    for index in range(ROWS):
        record = df.loc[index]
        author = record.author_name.strip().lower().replace(" ", "_")
        author_weight_matrix[index, author_feature_dict[author]] = author_weight  

    return csr_matrix(author_weight_matrix)


    
@profile
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

@profile
def get_desc_count_matrix(df):
    cv = CountVectorizer(stop_words="english")
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

@profile
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

similar_books = list(enumerate(cosine_similarity_matrix[book_ID]))
sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

limit = 6

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

random_books = np.random.randint(0, df.shape[0], 10)
random_books_cover = df.iloc[random_books].cover.values

finish_time = time.perf_counter()
passed_time = finish_time - start_time