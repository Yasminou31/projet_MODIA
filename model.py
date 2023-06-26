import torch

# Create NCF model
class NeuMF(torch.nn.Module):
    def __init__(self, num_users, num_recipes, embedding_dim, hidden_dim):
        super(NeuMF, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = torch.nn.Embedding(num_recipes, embedding_dim)
        
        self.mlp_user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.mlp_recipe_embedding = torch.nn.Embedding(num_recipes, embedding_dim)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
        self.output_layer = torch.nn.Linear(embedding_dim + hidden_dim, 1)
    
    def forward(self, user, recipe):
        mf_user_embedding = self.user_embedding(user)
        mf_recipe_embedding = self.recipe_embedding(recipe)
        mf_output = mf_user_embedding * mf_recipe_embedding
        
        mlp_user_embedding = self.mlp_user_embedding(user)
        mlp_recipe_embedding = self.mlp_recipe_embedding(recipe)
        mlp_input = torch.cat((mlp_user_embedding, mlp_recipe_embedding), dim=1)
        mlp_output = self.mlp(mlp_input)
        
        fusion_output = torch.cat((mf_output, mlp_output), dim=1)
        output = self.output_layer(fusion_output)
        
        return output.squeeze()