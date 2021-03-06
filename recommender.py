# Movie Recommender System
# Topic 
# COMP9417 - Project
# Written by Joel Lawrence (3331029) and Deepansh Singh (z5199370)
# Last modified, August '19

# Import required python libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import mean_squared_error as mse
import math, requests, json, shutil
import matplotlib.pyplot as plt

class recommender:
    def __init__(self):
        self.name = "Movie Recommender System by Joel and Deepansh"
        self.ratings_names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
        self.movies_names = ['Movie_ID', 'Title', 'Genres']
        self.link_names = ['Movie_ID', 'IMDB', 'MovieDB']
        self.url_names = ['Movie_ID', 'URL']
        return
    
    # Create user ratings into a dataframe
    def populate_user_ratings(self, filePath):
        self.ratingsDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.ratings_names)
        self.num_users = max(self.ratingsDF.User_ID)
        self.num_movies = max(self.ratingsDF.Movie_ID)
    
    # Create movie names into a dataframe
    def populate_movie_names(self, filePath):
        self.moviesDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.movies_names)
    
    # Create movie urls into a dataframe
    def populate_movie_url_images(self, filePath):
        self.imgURLsDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.url_names)
    
    # Create the links dataframe
    def populate_links(self, filePath):
        self.linksDF = pd.read_csv(filePath, skiprows=1, sep=',', names=self.link_names)
        self.linksDF.dropna(inplace=True)
        self.linksDF.MovieDB = self.linksDF.MovieDB.astype(int)

    # Create Ratings Matrix
    def initialise(self):
        self.ratings_matrix = np.zeros((self.num_users, self.num_movies))
        self.entries_counter = 0
        for rating in self.ratingsDF.itertuples():
            self.ratings_matrix[rating[1]-1, rating[2]-1] = rating[3]
            self.entries_counter += 1
        print("Ratings matrix has been built")
    
    # Clean data and build training and test sets
    def data_processing(self, percentage):
        self.sparsity = float(self.entries_counter)
        self.size = self.num_users * self.num_movies
        self.sparsity /= self.size
        self.test_item_number = math.floor(percentage * self.sparsity * self.num_movies)
        print(str(self.test_item_number) + ' ratings for each user are selected as testing dataset.') 
        self.training_set = self.ratings_matrix.copy()
        self.testing_set = np.zeros((self.num_users, self.num_movies))
        for uid in range(self.num_users):
            item = np.random.choice(self.ratings_matrix[uid, :].nonzero()[0], size=self.test_item_number, replace=False)
            self.testing_set[uid, item] = self.ratings_matrix[uid, item]
            self.training_set[uid, item] = 0
        print("Data has been processed.")

    # Calculate cosine similarity on training set
    def calc_similarity(self):
        self.user_similarity = cos_sim(self.training_set)
        print('User based similarity matrix built...')

    # Prediction using all users for similarity
    def prediction_using_all_users(self):
        # Denominator is the sum of similarity for each user with all other users.
        denom = np.array([np.abs(self.user_similarity).sum(axis=1)]).T
        
        # Numerator is the sum of similarity of user and other users * the ratings given by other users
        numer = self.user_similarity.dot(self.training_set)
        prediction_matrix = numer / denom
        print('Prediction based on all users similarity is done...')

        # get the real values which are not zero in test data set.
        true_values = self.testing_set[self.testing_set.nonzero()].flatten()

        # get the predicted values of those which are not zero in test data set.
        self.predicted_values_all = prediction_matrix[self.testing_set.nonzero()].flatten()
        
        # calculate mean squared error of results
        error = mse(self.predicted_values_all, true_values)
        print('The mean squared error of user_based CF is: ' + str(error))
        return error
    
    # Prediction method using the Top-K Neighbours
    def prediction_using_finite_nearest_neighbours(self, num_neighbours):
        prediction_matrix = np.zeros(self.testing_set.shape)
        
        for user in range(self.user_similarity.shape[0]):
            # exclude the get the top num_neighbours users' indexes other than user itself        
            index_top_neighbour = [np.argsort(self.user_similarity[:,user])[-2:-num_neighbours-2:-1]]
            
            for item in range(self.training_set.shape[1]):
                # Denominator is the sum of similarity for each user with its top k users:
                denom = np.sum(self.user_similarity[user,slice(None)][index_top_neighbour])        
                # Numerator
                numer = self.user_similarity[user,slice(None)][index_top_neighbour].dot(self.training_set[slice(None),item][index_top_neighbour])                
                prediction_matrix[user, item] = numer/denom
        print('Prediction based on top-' + str(num_neighbours) + ' users similarity is done...')        
        
        true_values = self.testing_set[self.testing_set.nonzero()].flatten()
        
        # get the predicted values of those which are not zero in test data set.
        predicted_values = prediction_matrix[self.testing_set.nonzero()].flatten()
        
        # 5.3 calculate MSE
        error = mse(predicted_values, true_values)
        print('The mean squared error of top-' + str(num_neighbours) + ' user_based CF is: ' + str(error) + '\n')
        return error
    
    # Method to predict a list of recommended movies using Top 30 most similar users
    def rating_recommender(self, user):
        similarity_matrix = cos_sim(self.ratings_matrix)
        prediction_matrix = np.zeros(self.ratings_matrix.shape)
        index_top30 = [np.argsort(similarity_matrix[:,user])[-2:-30-2:-1]]
        for item in range(self.rating_matrix.shape[1]):
            if self.rating_matrix[user][item] == 0:
                # Denominator is the sum of similarity for each user with its top 30 users:
                denom = np.sum(similarity_matrix[user,:][index_top30])
                
                # Numerator
                numer = similarity_matrix[user,:][index_top30].dot(self.rating_matrix[:,item][index_top30])
                
                prediction_matrix[user, item] = numer/denom
                    
        movie_ids = [i for i in np.argsort(prediction_matrix[user, :])[-30:]]
        return movie_ids

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
    
    # function to create a CSV containing all the links for each movie id
    def create_imageURL_csv(self):
        counter = 0
        with open("data/imgURLs.csv", 'w') as f:
            for row in self.linksDF.itertuples():
                movie_id = str(row[1])
                movieDB = str(int(row[3]))
                if movieDB == None:
                    continue
                URL = self.get_image_URL(movieDB)
                if URL == None:
                    line = movie_id + ',' + "missing" + '\n'
                else:
                    line = movie_id + ',' + URL + '\n'
                print(line)
                f.write(line)
                counter += 1
                if (counter % 100 == 0):
                    print("100 down")
        return

if __name__ == '__main__':
    recommender = recommender()
    recommender.populate_user_ratings("data/ratings.csv")
    recommender.populate_movie_names("data/movies.csv")
    recommender.initialise()
    recommender.data_processing(0.1)
    recommender.calc_similarity()
    all_error = recommender.prediction_using_all_users()
    # reduce the size of this list if you need to bring down program runtime.....
    sample_neighbours_numbers = [25, 30, 35, 40, 45, 50]
    errors = []
    for _ in sample_neighbours_numbers:
        error = recommender.prediction_using_finite_nearest_neighbours(_)
        errors.append(error)
    sample_neighbours_numbers.append(recommender.num_users)
    errors.append(all_error)
    y_pos = np.arange(len(sample_neighbours_numbers))     
    plt.bar(y_pos, errors, align='center', alpha=0.5)
    plt.xticks(y_pos, sample_neighbours_numbers)
    plt.ylabel('MSE')
    plt.title('Testing MSEs with varied k values')
    plt.savefig("output.png")
    plt.show() 