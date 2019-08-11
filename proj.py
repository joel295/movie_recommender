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
from math import sqrt
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



class Learner:
    def __init__(self,clf):
        self.clf=clf
    
    def scoring(self,X_train,y_train,pipe_lr):
        kfold = StratifiedKFold(y=y_train, n_folds=4 , random_state=1) #make 4 fold validation
        scores=[]
        for k, (train,test) in enumerate(kfold): #run on each fold and output the accuracy score
            pipe_lr.fit(X_train[train],y_train[train])
            score=pipe_lr.score(X_train[train],y_train[train])
            scores.append(score)
            print('Fold %s, acc %.3f' % (k+1,score))
    
        print("CROSS VALIDATION ACCURACY: %.3f +/- %.3f" % (np.mean(scores),np.std(scores))) #final accuracy of the classifier
    
    
    def data_processing(self):
        rating_df = pd.read_csv('data/ratings.csv') #read ratings.csv
        users = rating_df['userId'].unique() #take unique user_id values
        movies = rating_df['movieId'].unique() #take unique movie_id values
        user2idx = {k: i for i, k in enumerate(users)} 
        movie2idx = {k: i for i, k in enumerate(movies)}
        #combine them into a single dataframe
        rating_df['userId'] = rating_df['userId'].apply(lambda x: user2idx[x]) 
        rating_df['movieId'] = rating_df['movieId'].apply(lambda x: movie2idx[x])
        df = pd.read_csv('data/tags.csv')
        tags = df['tag'].unique() #take unique tags
        tag2idx = {k: i for i,k in enumerate(tags)}
        #add it to the dataframe
        rating_df['tag'] = df['tag'].apply(lambda x: tag2idx[x])
        #replace NaN values
        rating_df['tag'].fillna(-1,inplace=True)
        rating_df.drop('timestamp',axis=1,inplace=True) #drop timestamp because we dont need it
        #print(rating_df)
        Y = rating_df.iloc[:,-2].values #split the dataframe into X,Y . Ratings are put in Y
        rating = rating_df.pop('rating')
        #uder_id= rating_df.pop('userId')
        X = rating_df.iloc[:,:].values #rest of the features are put into X
        X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=0.3, random_state=0) #split the total dataset into 70% train and 30% test
        
        lab_enc = preprocessing.LabelEncoder() #to replace continous values
        y_train = lab_enc.fit_transform(y_train)
        y_test = lab_enc.fit_transform(y_test)
        return X_train, X_test, y_train, y_test
    
    
    def classifier(self,X_train, X_test, y_train, y_test):
        # Regression classifier
        # Outputs training accuract
        # Output the predictions it predicts
        # Low Accuracy due to lack of features
        if self.clf == "regression": 
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            print("LogisticRegression")
            print("Training accuracy: ",clf.score(X_train,y_train)*100)
            #print("predictions: ",clf.predict(X_test))

        # KNN classifier
        # Outputs training accuract
        # Output the predictions it predicts
        # 50% Accuracy, highest among all
        elif self.clf == "knn":
            clf = KNeighborsClassifier(n_neighbors=10)
            clf.fit(X_train, y_train)
            print("knn")
            print("Training accuracy: ",clf.score(X_train,y_train)*100)
            #print("predictions: ",clf.predict(X_test))

        # SVC classifier
        # Outputs training accuract
        # Output the predictions it predicts
        # Low Accuracy due to lack of features
        elif self.clf == "svc":        
            clf = SVC()
            clf.fit(X_train, y_train)
            print("SVC")
            print("Training accuracy: ",round(clf.score(X_train,y_train)*100,2))
            #print("predictions: ",clf.predict(X_test))

        # Random Forest classifier
        # Outputs training accuract
        # Output the predictions it predicts
        # Over fitting data due to lack of features
        # not useful
        elif self.clf == "random forest":
            clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1)
            clf.fit(X_train, y_train)
            print("Random Forest")
            print("Training accuracy: ",round(clf.score(X_train,y_train)*100,2))
            #print("predictions: ",clf.predict(X_test))

        # Neural Nets 
        # Outputs training accuract
        # Output the predictions it predicts
        # Low Accuracy due to lack of features

        elif self.clf == "neural nets":
            clf = MLPClassifier(solver='lbfgs', alpha = 1e-5 , hidden_layer_sizes=(10,5), random_state=1)
            clf.fit(X_train, y_train)
            print("neural nets")
            print("Training accuracy: ",round(clf.score(X_train,y_train)*100,2))
            #print("Predictions: ",clf.predict(X_test))
            
        elif self.clf == "gnb":
            gaussian = GaussianNB()
            gaussian.fit(X_train, y_train)
            #Y_pred = gaussian.predict(X_test)
            acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
            print("gaussian NB\n")
            print("Training accuracy: ",acc_gaussian)
        
        elif self.clf == "dtree":
            decision_tree = DecisionTreeClassifier()
            decision_tree.fit(X_train, y_train)
            #Y_pred = decision_tree.predict(test)
            accuracy = round(decision_tree.score(X_train, y_train) * 100, 2)
            print("Decision Tree\n")
            print("Training accuracy: ",accuracy)
            
        elif self.clf == "sgd":
            sgd = SGDClassifier()
            sgd.fit(X_train, y_train)
            accuracy = round(sgd.score(X_train, y_train) * 100, 2)
            print("Stochastic Gradient Descent")
            print("Training accuracy: ",accuracy)

    
    
    def pipeline_learning(self,X_train, X_test, y_train, y_test):
        #pipeline does the same work as normal classifiers, ive not standardised the data and have not used PCA because 
        # there are very less features already.
        # PIPELINING does apply k-fold validation on the dataset and outputs the accuracy at each fold and overall training accuracy
        # most of the classifiers gave just a slight increase in the accuracy but nothing substantial

        if self.clf == "knn":
            #Pipeline code
            print('KNN\n')
            pipe_lr= Pipeline([('clf',KNeighborsClassifier(n_neighbors=4))])
            self.scoring(X_train,y_train,pipe_lr)
            
        elif self.clf == "neural nets":
            print("Neural Nets")
            pipe_lr= Pipeline([('clf',MLPClassifier(solver='lbfgs', alpha = 1e-5 , hidden_layer_sizes=(10,5), random_state=1))])
            self.scoring(X_train,y_train,pipe_lr)
            
            
        elif self.clf == "random forest":
            print('RANDOM FOREST\n')
            #random = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1))
            pipe_lr= Pipeline([('clf',RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1))])
            self.scoring(X_train,y_train,pipe_lr)
            
        elif self.clf == "regression":
            print('LOGISTIC REGRESSION\n')
            pipe_lr= Pipeline([('clf',LogisticRegression(random_state=1))])
            self.scoring(X_train,y_train,pipe_lr)
            
        elif self.clf == "svc":
            print('SVC\n')
            pipe_svc = Pipeline([('clf',SVC(random_state=1))])
            self.scoring(X_train,y_train,pipe_svc)
    
        elif self.clf == "gnb":
            pipe_lr= Pipeline([('clf',GaussianNB())])
            print("gaussian NB\n")
            self.scoring(X_train,y_train,pipe_lr)

        
        elif self.clf == "dtree":
            pipe_lr= Pipeline([('clf',DecisionTreeClassifier())])
            print("Decision Tree\n")
            self.scoring(X_train,y_train,pipe_lr)
 
        elif self.clf == "sgd":
            pipe_lr= Pipeline([('clf',SGDClassifier())])
            print("Stochastic Gradient Descent")
            self.scoring(X_train,y_train,pipe_lr)



    def create_similarity_matrix(self):
        rr = pd.read_csv('data/ratings.csv')
        x = rr['userId'].max() #max userID
        y = rr['movieId'].max() #max movie_id
        rating1=np.zeros((x,y)) #create a x,y matrix full of zeros
        total_ratings=0
        for  row in rr.itertuples():
            rating1[row[1]-1,row[2]-1]=row[3] #put the available ratings
            total_ratings+=1  #add the total number of actual rating in the matrix
        
        how_sparse=float(total_ratings)
        size=x*y #total entreies
        sparse = how_sparse/size #calculate sparsity
        test_item = math.floor(0.1*sparse*y)
        train_set = rating1.copy()
        test_set = np.zeros((x,y))
        #dividing it into training and test set. fill zeros where there was rating to create a random set.
        for uid in range(x):
                item = np.random.choice(rating1[uid, :].nonzero()[0], size=test_item, replace=False)
                test_set[uid, item] = rating1[uid, item] #add to the test_set
                train_set[uid, item] = 0                
        user_sim = cos_sim(train_set) #calculating the cosine similarity
        return rating1,user_sim      

    def pair_wise_distances(self,rating_matrix):
        cos_similarity  = 1-pairwise_distances(rating_matrix, metric="cosine") #calculates similarity on the basis of cosine metric
        pearson_similarity  = 1-pairwise_distances(rating_matrix, metric = "correlation") #calculates similarity on the basis of correlation metric
        
    def find_similar_users(self,user_id, ratings, metric, k=4):
        similar=[]
        index=[]
        model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')  #create model
        model_knn.fit(ratings) #fit ratings ( user_id x movie_id)
        distances, index = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1) #find the distances
        similar = 1-distances.flatten() #find similar users
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
        ratings=ratings.T # transpose the matrix so as to get movie_id x useer_id
        movie_df = pd.read_csv('data/movies.csv') #read the movies_csv file to get the movie_id and title names
        genre = movie_df.pop('genres') #remove genres as we dont need it
        model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')  #create the model
        model_knn.fit(ratings) # fit the rating matrix
        distances, index = model_knn.kneighbors(ratings.iloc[movie_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
        similar = 1-distances.flatten() #find similar movies
        if any(movie_df.movieId == movie_id):
            movie_nam = movie_df.loc[movie_df['movieId']==movie_id,'title'].iloc[0]
        else:
            #print("movie_id doesnt exist in the database\n")
            movie_nam = str(movie_id)
        print("%2d most similar movies for Movie %s:\n"%(k,movie_nam)) #print the movie name
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == movie_id:
                continue
            else:
                if any(movie_df.movieId == index.flatten()[i]+1):
                    movie_name=movie_df.loc[movie_df['movieId']==(index.flatten()[i]+1),'title'].iloc[0]
                else:
                    movie_name = index.flatten()[i]+1
                print("%2d: %s, with similarity of %3.5f"%(i, movie_name , similar.flatten()[i])) #print the similar movie names
        return similar,index
    
    
    def predict_userbased_rating(self,user_id, movie_id, ratings, metric, k=4):
        prediction=0
        similar, index=self.find_similar_users(user_id, ratings,metric, k) #similar users based on cosine similarity
        mean_rating = ratings.loc[user_id-1,:].mean() #to adjust for zero based indexing
        sum_wt = np.sum(similar)-1 #calculate the weight
        product=1
        wtd_sum = 0 
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == user_id:
                continue
            else: 
                ratings_diff = ratings.iloc[index.flatten()[i],movie_id-1]-np.mean(ratings.iloc[index.flatten()[i],:]) #calculate the difference in rating
                product = ratings_diff * (similar[i])  #account the difference in rating
                wtd_sum = wtd_sum + product #update wt
        
        prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
        print('\nPredicted rating for user %2d -> movie %2d: %2d'%(user_id,movie_id,prediction))
        return prediction
    
    def predict_moviebased_rating(self,user_id, movie_id, ratings, metric, k=4):
        #very similar function to predict user based ratings, just the matrux provded to it is transposed
        prediction=0
        wtd_sum=0
        similar, index=self.find_similar_movies(movie_id, ratings,metric,k) #similar users based on cosine similarity
        sum_wt = np.sum(similar)-1 
        product=1
        for i in range(0, len(index.flatten())):
            if index.flatten()[i]+1 == movie_id:
                continue
            else: 
                product =  ratings.iloc[user_id-1,index.flatten()[i]] * (similar[i])
                wtd_sum = wtd_sum + product
                
        prediction = int(round((wtd_sum/sum_wt)))
        print('\nPredicted rating for user %2d -> movie %2d: %2d'%(user_id,movie_id,prediction))
        return prediction
    
    
    def RMSE(self, ratings , metric):
        
        n_users = ratings.shape[0]
        n_items = ratings.shape[1]
        prediction = np.zeros((n_users, n_items))
        prediction= pd.DataFrame(prediction)
        
        for i in range(n_users):
            for j in range(n_items):
                prediction[i][j] = self.predict_userbased_rating(i+1, j+1, ratings, metric)
                
        MSE = mean_squared_error(prediction, ratings)
        RMSE = round(sqrt(MSE),3)
        print("RMSE using item based approach with %s metric is: %5.5f"%(metric,RMSE))        

        prediction = np.zeros((n_users, n_items))
        prediction= pd.DataFrame(prediction)
        
        for i in range(n_users):
            for j in range(n_items):
                prediction[i][j] = self.predict_moviebased_rating(i+1, j+1, ratings, metric)
      
        MSE = mean_squared_error(prediction, ratings)
        RMSE = round(sqrt(MSE),3)
        print("RMSE using item based approach with %s metric is: %5.5f"%(metric,RMSE))
                  
    
    
if __name__ == '__main__':
    #different learners used are:remove the comment from clfs
    print("Different classifiers running without k-fold validation and with 4-fold validation")
    #clfs = ['knn','neural nets', 'random forest', 'regression', 'gnb' , 'dtree' , 'sgd'] 
    clfs = ['knn']
    
    for clf in clfs: #clfs store the name of all the classifiers tested, but currently it is set to knn
        learner = Learner(clf)
        X_train, X_test, y_train, y_test = learner.data_processing()
        learner.classifier(X_train, X_test, y_train, y_test )
        learner.pipeline_learning(X_train, X_test, y_train, y_test)
    
    #learner = Learner('knn')
    print('~~~~~~~~~~~~xx~~~~~~~~~~~~~~~')
    rating_matrix,user_similarity = learner.create_similarity_matrix()
    rating_matrix = pd.DataFrame(rating_matrix)
    user_id = int(input("Enter the user id: "))
    movie_id = int(input("Enter the movie_id: "))
    metric = input("Enter the metric - cosine or correlation: ")
    print('\n')
    print("USER BASED SIMILARITY\n")
    prediction_userbased = learner.predict_userbased_rating(user_id,movie_id,rating_matrix, metric = metric)
    print("ITEM BASED SIMILARITY\n")
    prediction_moviebased = learner.predict_moviebased_rating(user_id,movie_id,rating_matrix, metric = metric)
    #eval = learner.RMSE(rating_matrix, metric)