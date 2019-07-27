import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

class recommender:
    def __init__(self):
        self.name = "Movie Recommender System by Joel and Deepansh"
        self.ratings_names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
        return
    
    def populate_user_ratings(self, filePath):
        self.ratingsDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.ratings_names)
        
        self.num_users = max(self.ratingsDF.User_ID)
        self.num_movies = max(self.ratingsDF.Movie_ID)
        print(str(self.num_users) + ' Users in total.' + '\n') 
        print(str(self.num_movies) + ' Movies in total.' + '\n')
        
        
        


if __name__ == '__main__':
    recommender = recommender()
    recommender.populate_user_ratings("data/ratings.csv")