# Movie Recommender System
# Topic ####
# COMP9417 - Project
# Written by Joel Lawrence (3331029) and Deepansh Singh ()
# Last modified, July '19

import pandas as pd
import numpy as np
import sklearn as sk
import math

class recommender:
    def __init__(self):
        self.name = "Movie Recommender System by Joel and Deepansh"
        self.ratings_names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
        return
    
    def populate_user_ratings(self, filePath):
        self.ratingsDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.ratings_names)
        self.num_users = max(self.ratingsDF.User_ID)
        self.num_movies = max(self.ratingsDF.Movie_ID)
        print("Users in total: %d\nMovies in total: %d\n" % (self.num_users, self.num_movies))
        
    def initialise(self):
        ratings_matrix = np.zeros((self.num_users, self.num_movies))
        self.entries_counter = 0
        for rating in self.ratingsDF.itertuples():
            ratings_matrix[rating[1]-1, rating[2]-1] = rating[3]
            self.entries_counter += 1
    
    ## curation to training and test set here

        


if __name__ == '__main__':
    recommender = recommender()
    recommender.populate_user_ratings("data/ratings.csv")
    recommender.initialise()