# Movie Recommender System
# Topic ####
# COMP9417 - Project
# Written by Joel Lawrence (3331029) and Deepansh Singh ()
# Last modified, July '19

import pandas as pd
import numpy as np
import sklearn as sk
import math, requests, json, shutil

class recommender:
    def __init__(self):
        self.name = "Movie Recommender System by Joel and Deepansh"
        self.ratings_names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
        self.movies_names = ['Movie_ID', 'Title', 'Genres']
        self.link_names = ['Movie_ID', 'IMDB', 'MovieDB']
        return
    
    def populate_user_ratings(self, filePath):
        self.ratingsDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.ratings_names)
        self.num_users = len(set(self.ratingsDF.User_ID))
        self.num_movies = len(set(self.ratingsDF.Movie_ID))
        print("Users in total: %d\nMovies in total: %d\n" % (self.num_users, self.num_movies))
    
    def populate_movie_names(self, filePath):
        self.moviesDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.movies_names)
        print("Movies in total: %d\n" % (len(set(self.moviesDF.Movie_ID))))
    
    def populate_links(self, filePath):
        self.linksDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.link_names)

    def initialise(self):
        ratings_matrix = np.zeros((self.num_users, self.num_movies))
        self.entries_counter = 0
        for rating in self.ratingsDF.itertuples():
            ratings_matrix[rating[1]-1, rating[2]-1] = rating[3]
            self.entries_counter += 1
    
    # function to get the URL of a movie poster using MovieDB id
    def get_image_URL(self, movie_id):
        baseURL = "https://image.tmdb.org/t/p/"
        file_size = "original"
        URL = "http://api.themoviedb.org/3/movie/" + movie_id + "?language=en-US&api_key=da0f779a51e5a27916afccf8a6ee84c2"
        r = requests.get(URL)
        if r.status_code != 200:
            del r
            return
        data = r.json()
        file_path = data["poster_path"]
        if file_path == None:
            return "None"
        file_type = file_path.split('.')[1]
        del r
        imgURL = baseURL + file_size + file_path
        return imgURL

    def create_imageURL_csv(self):
        #with open("data/imgURLs.csv", 'w') as f:
        #for index, row in self.linksDF.iterrows():
        #    movieID = str(row['Movie_ID'])
        #    movieDBID = row['MovieDB']
        #    print(movieDBID)

            # URL = self.get_image_URL(movieDBID)
            # if URL == None:
            #     line = movieID + ',' + "missing" + '\n'
            # else:
            #     line = movieID + ',' + URL + '\n'
            # print(line)
                # f.write(line)
        return

    ## curation to training and test set here


if __name__ == '__main__':
    recommender = recommender()
    recommender.populate_user_ratings("data/ratings.csv")
    recommender.populate_movie_names("data/movies.csv")
    recommender.populate_links("data/links.csv")
    recommender.create_imageURL_csv()