import torch

n_unique_users_train = 15318
n_unique_recipes_train = 12897

# Create NCF model
class NCF(torch.nn.Module):
    def __init__(self, n_users=n_unique_users_train, n_recipes=n_unique_recipes_train, n_factors=8):
        super().__init__()
        self.user_embeddings = torch.nn.Embedding(n_users, n_factors)
        self.recipe_embeddings = torch.nn.Embedding(n_recipes, n_factors)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_factors*2, out_features=64),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Linear(in_features=32, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, user, recipe):
        user_embedding = self.user_embeddings(user)
        recipe_embedding = self.recipe_embeddings(recipe)

        # Concatenate the two embedding layers
        z = torch.cat([user_embedding, recipe_embedding], dim=-1)
        return self.predictor(z)