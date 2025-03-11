# RECOMMENDATION-SYSTEM
"COMPANY" : CODTECH IT SOLUTIONS 
"NAME" : SRIMURUGAN 
"INTERN ID" : CT08VRH 
"DOMAIN" : MACHINE LEARNING
"DURATION" : 4 WEEKS
"MENTOR" : Muzammil Ahmed


Description of the Code: Recommendation System Using SVD
This Python script implements a simple recommendation system using Singular Value Decomposition (SVD) from the Surprise library. The goal is to predict user preferences based on historical ratings and suggest the top-rated items for a given user.

Key Components:
Importing Required Libraries

numpy and pandas for handling data.
surprise for building and evaluating the recommendation model.
Dataset Preparation

A dictionary ratings_dict is created containing user IDs, item IDs, and corresponding ratings.
The dataset is converted into a Pandas DataFrame.
The Reader class from Surprise is used to define the rating scale.
The data is then loaded into the Surprise Dataset format.
Building the Model

The dataset is split into a training set and a test set.
SVD (Singular Value Decomposition) is applied to the training data to learn user-item interactions.
Model Evaluation

The trained model is tested on the test set.
The Root Mean Square Error (RMSE) is calculated to measure prediction accuracy.
Generating Recommendations

A function get_top_n_recommendations is defined to suggest top-N items for a given user.
It predicts ratings for all items and sorts them in descending order.
The top-N highest-rated items are returned as recommendations.
Example Usage

The script calls the get_top_n_recommendations function for User 1, retrieving and displaying the top 3 recommended items.


OUTPUT :  ![Image](https://github.com/user-attachments/assets/3c0d24c6-84ba-4cf0-946a-21766b85f00d)
