# README 



# DATABASE

As a database, MongoDB is used. First, fetch the data from kaggle which is a cleaned version of scifi books dataset. Additionally, covers for each book is added to the dataset. Second, download the mongoimport utility from MongoDB website. Third, assuming current prompt at project root run the following commands for uploading dataset to MongoDB atlas cluster (a free account is necessary):

```bash
$ cd data
$ mongoimport --uri mongodb+srv://<username>:<password>@cluster0.o3o9q.mongodb.net/<databasename> --collection BOOKS --type csv --headerline --file scifi_with_cover.csv
```

If DNS error occurs, `/etc/resolv.conf` needs to be modified. Add `8.8.8.8` to the file then try to import again.

