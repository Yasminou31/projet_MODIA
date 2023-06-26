import torch
import argparse
import pandas as pd
import pickle
from model import NeuMF

# load n_unique_users and n_unique_recipes from n_users_recipes.json
import json
with open('n_users_recipes.json', 'r') as f:
    n_users_recipes = json.load(f)
    n_unique_users = n_users_recipes['n_users']
    n_unique_recipes = n_users_recipes['n_recipes']

embedding_dim = 64
hidden_dim = 64

# define function to load model
def load_model(path):
    model = NeuMF(n_unique_users, n_unique_recipes, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(path))
    return model

# define function to predict
def predict(user_id, recipe_id, model):
    user = torch.tensor([user_id], dtype=torch.long)
    recipe = torch.tensor([recipe_id], dtype=torch.long)
    rating = model(user, recipe)
    return torch.round(rating).item()

if __name__ == '__main__':
    # parse arguments and get the model weights path and the path of the test_script.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='weights.pth')
    parser.add_argument('--test_script_path', type=str, default='test_script.csv')
    args = parser.parse_args()

    # get the model weights path and the path of the test_script.csv
    model_path = args.model_path
    test_script_path = args.test_script_path

    # load model
    model = load_model(model_path)

    # load the user2id and recipe2id dictionaries using pickle
    user2id = pickle.load(open('user2id.pkl', 'rb'))
    recipe2id = pickle.load(open('recipe2id.pkl', 'rb'))

    # load test_script.csv
    test_script = pd.read_csv(test_script_path)
    
    # for each row in test_script.csv, predict the rating and print the predictions next to the real label
    for index, row in test_script.iterrows():
        user_id = user2id[row['user_id']]
        recipe_id = recipe2id[row['recipe_id']]
        rating = row['rating']
        prediction = predict(user_id, recipe_id, model)
        print(f'User: {user_id} - Recipe: {recipe_id} - Rating: {rating} - Predicted rating: {prediction}')
