from Seqbacksel import SBS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import math
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine



class Learner:
    def __init__(self,clf):
        self.clf=clf
    
    def scoring(self,X_train,y_train,pipe_lr):
        kfold = StratifiedKFold(y=y_train, n_folds=3 , random_state=1)
        scores=[]
        for k, (train,test) in enumerate(kfold):
            pipe_lr.fit(X_train[train],y_train[train])
            score=pipe_lr.score(X_train[train],y_train[train])
            scores.append(score)
            print('Fold %s,class dist %s, acc %.3f' % (k+1,np.bincount(y_train[train]),score))
    
        print("CROSS VALIDATION ACCURACY: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))
    
    
    def data_processing(self):
        rating_df = pd.read_csv('data/ratings.csv')
        users = rating_df['userId'].unique()
        movies = rating_df['movieId'].unique()
        user2idx = {k: i for i, k in enumerate(users)}
        movie2idx = {k: i for i, k in enumerate(movies)}
        rating_df['userId'] = rating_df['userId'].apply(lambda x: user2idx[x])
        rating_df['movieId'] = rating_df['movieId'].apply(lambda x: movie2idx[x])
        df = pd.read_csv('data/tags.csv')
        tags = df['tag'].unique()
        tag2idx = {k: i for i,k in enumerate(tags)}
        rating_df['tag'] = df['tag'].apply(lambda x: tag2idx[x])
        rating_df['tag'].fillna(-1,inplace=True)
        rating_df.drop('timestamp',axis=1,inplace=True)
        #print(rating_df)
        Y = rating_df.iloc[:,-2].values
        rating = rating_df.pop('rating')
        #uder_id= rating_df.pop('userId')
        X = rating_df.iloc[:,:].values
        X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=0.3, random_state=0)
        
        lab_enc = preprocessing.LabelEncoder()
        y_train = lab_enc.fit_transform(y_train)
        y_test = lab_enc.fit_transform(y_test)
        return X_train, X_test, y_train, y_test
    
    
    def classifier(self,X_train, X_test, y_train, y_test):
        if self.clf == "regression":
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            print("LogisticRegression")
            print(clf.score(X_train,y_train))
            print(clf.predict(X_test))
        elif self.clf == "knn":
            clf = KNeighborsClassifier(n_neighbors=10)
            clf.fit(X_train, y_train)
            print("knn")
            print(clf.score(X_train,y_train))
            print(clf.predict(X_test))
        elif self.clf == "svc":        
            clf = SVC(random_state=1)
            clf.fit(X_train, y_train)
            print("SVC")
            print(clf.score(X_train,y_train))
            print(clf.predict(X_test))
        elif self.clf == "random forest":
            clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1)
            clf.fit(X_train, y_train)
            print("Random Forest")
            print(clf.score(X_train,y_train))
            print(clf.predict(X_test))
        elif self.clf == "Neural Nets":
            clf = MLPClassifier(solver='lbfgs', alpha = 1e-5 , hidden_layer_sizes=(20,2), random_state=1)
            clf.fit(X_train, y_train)
            print("neural nets")
            print(clf.score(X_train,y_train))
            print(clf.predict(X_test))
    
    
    def pipeline_learning(self,X_train, X_test, y_train, y_test):
        if self.clf == "knn":
            #Pipeline code
            print('KNN\n')
            pipe_lr= Pipeline([('clf',KNeighborsClassifier(n_neighbors=4))])
            self.scoring(X_train,y_train,pipe_lr)
            
        elif self.clf == "neural nets":
            print("Neural Nets")
            pipe_lr= Pipeline([('clf',MLPClassifier(solver='adam', alpha = 1e-5 , hidden_layer_sizes=(20,2), random_state=1))])
            self.scoring(X_train,y_train,pipe_lr)
            
            
        elif self.clf == "random forest":
            print('RANDOM FOREST\n')
            #random = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1))
            pipe_lr= Pipeline([('clf',RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1))])
            self.scoring(X_train,y_train,pipe_lr)
            
        elif self.clf == "regression":
            print('LOGISTIC REGRESSION\n')
            pipe_lr= Pipeline([('clf',LogisticRegression(random_state=1))])
            self.scoring(X_train,y_train,pipe_lr)
            
        elif self.clf == "svc":
            print('SVC\n')
            pipe_svc = Pipeline([('clf',SVC(random_state=1))])
            self.scoring(X_train,y_train,pipe_svc)
    
    def create_similarity_matrix(self):
        rr = pd.read_csv('data/ratings.csv')
        x = 610
        y =193609
        rating1=np.zeros((x,y))
        total_ratings=0
        for  row in rr.itertuples():
            rating1[row[1]-1,row[2]-1]=row[3]
            total_ratings+=1
        
        how_sparse=float(total_ratings)
        size=x*y
        sparse = how_sparse/size
        test_item = math.floor(0.1*sparse*y)
        train_set = rating1.copy()
        test_set = np.zeros((x,y))
        for uid in range(x):
                item = np.random.choice(rating1[uid, :].nonzero()[0], size=test_item, replace=False)
                test_set[uid, item] = rating1[uid, item] #add to the test_set
                train_set[uid, item] = 0
                
        user_sim = cos_sim(train_set)
        return rating1,user_sim      

    def pair_wise_distances(self,rating_matrix):
        cos_sim  = 1-pairwise_distances(rating_matrix, metric="cosine")
        pearson_similarity  = 1-pairwise_distances(rating_matrix, metric = "correlation")
        #print(pearson_similarity)
        
    def find_similar_users(self,user_id, ratings, metric, k=4):
        similar=[]
        index=[]
        model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
        model_knn.fit(ratings)
        distances, index = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
        similar = 1-distances.flatten()
        print("%2d most similar users for User %2d:\n"%(k,user_id))
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == user_id:
                continue
            else:
                print("%2d: User %2d, with similarity of %3.5f"%(i, index.flatten()[i]+1, similar.flatten()[i]))
        return similar,index
    
    
    def find_similar_movies(self,movie_id, ratings, metric, k=4):
        similar=[]
        index=[]
        model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
        model_knn.fit(ratings)
        distances, index = model_knn.kneighbors(ratings.iloc[movie_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
        similar = 1-distances.flatten()
        print("%2d most similar movies for Movie %2d:\n"%(k,movie_id))
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == movie_id:
                continue
            else:
                print("%2d: Movie %2d, with similarity of %3.5f"%(i, index.flatten()[i]+1, similar.flatten()[i]))
        return similar,index
    
    
    def predict_userbased_rating(self,user_id, movie_id, ratings, metric, k=4):
        prediction=0
        similar, index=self.find_similar_users(user_id, ratings,metric, k) #similar users based on cosine similarity
        mean_rating = ratings.loc[user_id-1,:].mean() #to adjust for zero based indexing
        sum_wt = np.sum(similar)-1
        product=1
        wtd_sum = 0 
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == user_id:
                continue
            else: 
                ratings_diff = ratings.iloc[index.flatten()[i],movie_id-1]-np.mean(ratings.iloc[index.flatten()[i],:])
                product = ratings_diff * (similar[i])
                wtd_sum = wtd_sum + product
        
        prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
        print('\nPredicted rating for user %2d -> item %2d: %2d'%(user_id,movie_id,prediction))
        return prediction
    
    def predict_moviebased_rating(self,user_id, movie_id, ratings, metric, k=4):
        prediction=0
        wtd_sum=0
        similar, index=self.find_similar_movies(movie_id, ratings,metric) #similar users based on cosine similarity
        sum_wt = np.sum(similar)-1
        mean_rating = ratings.loc[:,movie_id-1].mean()
        product=1
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == movie_id:
                continue
            else: 
                product =  ratings.iloc[user_id-1,index.flatten()[i]] * (similar[i])
                wtd_sum = wtd_sum + product
                
        prediction = int(round(mean_rating+(wtd_sum/sum_wt)))
        print('\nPredicted rating for user %2d -> item %2d: %2d'%(user_id,movie_id,prediction))
        return prediction
    
if __name__ == '__main__':
    learner = Learner("knn")
    X_train, X_test, y_train, y_test = learner.data_processing()
    #learner.classifier(X_train, X_test, y_train, y_test )
    #learner.pipeline_learning(X_train, X_test, y_train, y_test)
    rating_matrix,user_similarity = learner.create_similarity_matrix()
    #learner.pair_wise_distances(rating_matrix)
    rating_matrix = pd.DataFrame(rating_matrix)
    #print(rating_matrix)
    user_id = int(input("Enter the user id: "))
    #similar,index = learner.find_similar_users(user_id,rating_matrix, metric='cosine') #use correlation insteaf of cosine
    movie_id = int(input("Enter the movie_id: "))
    prediction_userbased = learner.predict_userbased_rating(user_id,movie_id,rating_matrix, metric ='cosine')
    #similar,index = learner.find_similar_movies(movie_id,rating_matrix,metric='cosine')
    prediction_moviebased = learner.predict_moviebased_rating(user_id,movie_id,rating_matrix, metric ='cosine')
