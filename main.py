import torch
from model import NCF

# define function to load model
def load_model(path):
    model = NCF()
    model.load_state_dict(torch.load(path))

# define function to predict
def predict(user_id, recipe_id, model):
    user = torch.tensor([user_id], dtype=torch.long)
    recipe = torch.tensor([recipe_id], dtype=torch.long)
    rating = model(user, recipe)
    return int(torch.round(rating * 5))