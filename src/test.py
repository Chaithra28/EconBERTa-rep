import torch
from src import config
from src.model import CRFTagger
from src.evaluation import get_validation_performance
from src.utils import analyze_generalization
from src.utils import seed_everything, label_dict, device
from src.data_preprocessing import read_conll, get_dataset
from transformers import AutoTokenizer

seed_everything()

assert torch.cuda.is_available()
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")

# Load tokenizer
model_name = config.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

# Load the pre-trained model
model = CRFTagger(config.model_name, len(label_dict))
model.dropout = torch.nn.Dropout(config.dropout_rate)
model.to(device)


# Read data
train_df = read_conll('data/econ_ie/train.conll')
val_df = read_conll('data/econ_ie/dev.conll')
test_df = read_conll('data/econ_ie/test.conll')

# Tokenize and format
_, train_sentences = get_dataset(train_df, tokenizer, label_dict)
test_set, test_sentences = get_dataset(test_df, tokenizer, label_dict)

lr = config.learning_rates[2]

# Load state_dict of the model
model.load_state_dict(torch.load(f'{config.model_name}_{lr}_best.pth'))

# Evaluate the model on the test set
print(get_validation_performance(test_set, model, device, label_dict, config.batch_size))
lengths, metrics = analyze_generalization(model, test_set, tokenizer, train_sentences)

