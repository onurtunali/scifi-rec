# README 

This is a book recommendation app deployable to free tier of Streamlit Cloud. A jupyter notebook version can be found in [kaggle](). The most prominent feature of this app is its ability to toggle weights of features used in similarity matrix. 

A common approach to recommendation methods is to vectorize a book description or extracted features and then feed them to a similarity measure such as cosine similarity. However, native APIs of conventional libraries are not able to adjust vector weights manually. 

# Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-black.svg?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) 

# Database

As a database, MongoDB is used. First, fetch the data from kaggle which is a cleaned version of scifi books dataset. Additionally, covers for each book is added to the dataset. Second, download the mongoimport utility from MongoDB website. Third, assuming current prompt at project root run the following commands for uploading dataset to MongoDB atlas cluster (a free account is necessary):

```bash
$ cd data
$ mongoimport --uri mongodb+srv://<username>:<password>@cluster0.o3o9q.mongodb.net/<databasename> --collection BOOKS --type csv --headerline --file scifi_with_cover.csv
```

If DNS error occurs, `/etc/resolv.conf` needs to be modified. Add `8.8.8.8` to the file then try to import again.

