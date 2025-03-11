import numpy as np
import pandas as pd

try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
except ModuleNotFoundError:
    print("The 'surprise' library is not installed. Install it using: pip install scikit-surprise")
    exit()

# Load dataset
ratings_dict = {
    "user": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    "item": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "rating": [5, 3, 4, 4, 2, 3, 3, 5, 4],
}
df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

# Build full trainset and testset
trainset = data.build_full_trainset()
testset = trainset.build_testset()

# Apply SVD for matrix factorization
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)

# Function to get top-N recommendations
def get_top_n_recommendations(model, user_id, n=3):
    item_ids = df['item'].unique()
    user_ratings = {iid: model.predict(user_id, iid).est for iid in item_ids}
    top_items = sorted(user_ratings, key=user_ratings.get, reverse=True)[:n]
    return top_items

# Example usage
user_id = 1
recommendations = get_top_n_recommendations(model, user_id)
print(f"Top recommendations for user {user_id}: {recommendations}")