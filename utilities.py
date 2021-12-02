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