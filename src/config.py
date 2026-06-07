import torch

class Config:
    def __init__(self):
        self.image_size = 64
        self.input_channels = 3
        self.output_channels = 3
        self.num_timesteps = 1000
        self.batch_size = 32
        self.learning_rate = 0.00008
        self.num_epochs = 201
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_path = "data"
        self.model_path = "models"
        self.log_path = "logs"
        self.checkpoint_path = "checkpoints"
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_hat = torch.sqrt(self.alphas_hat)
        self.sqrt_one_minus_alphas_hat = torch.sqrt(1 - self.alphas_hat)

config = Config()

