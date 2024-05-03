# src/config.py
from src.utils import ModelName

# Hyperparameters for data preprocessing
max_length = 128

# Hyperparameters for the model
model_name = ModelName.ECONBERTA_FC
dropout_rate = 0.2
learning_rates = [5e-5, 6e-5, 7e-5]
batch_size = 12
gradient_accumulation_steps = 4
weight_decay = 0
max_epochs = 10
lr_decay = "slanted_triangular"
fraction_of_steps = 0.06
adam_epsilon = 1e-8
adam_beta1 = 0.9
adam_beta2 = 0.999

