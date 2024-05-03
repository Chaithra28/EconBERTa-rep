import time
import torch
from src import config
from src.utils import   analyze_generalization, seed_everything, extract_model_name, label_dict, device
from src.model import CRFTagger
from src.data_preprocessing import read_conll, get_dataset
from src.evaluation import get_validation_performance
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

seed_everything()

assert torch.cuda.is_available()
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")

# Load tokenizer
model_name = config.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

# Read data
train_df = read_conll('data/econ_ie/train.conll')
val_df = read_conll('data/econ_ie/dev.conll')
test_df = read_conll('data/econ_ie/test.conll')

# Tokenize and format
train_set, train_sentences = get_dataset(train_df, tokenizer, label_dict)
val_set, val_sentences = get_dataset(val_df, tokenizer, label_dict)
test_set, test_sentences = get_dataset(test_df, tokenizer, label_dict)

# Load model
model = CRFTagger(model_name, len(label_dict))
model.dropout = torch.nn.Dropout(config.dropout_rate)
model.to(device)


# Calculate the total number of training steps
total_steps = (len(train_set) // (config.batch_size * config.gradient_accumulation_steps)) * config.max_epochs

# Train the model
for lr in config.learning_rates:
    print(f"Current learning rate: {lr}")

    # Create the optimizer with the specified hyperparameters
    optimizer = AdamW(model.parameters(), lr=lr, eps=config.adam_epsilon, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay, no_deprecation_warning=True)

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * config.fraction_of_steps), num_training_steps=total_steps)

    # Training loop
    for epoch_i in range(config.max_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config.max_epochs))
        print('Training...')

        total_train_loss = 0
        model.train()

        num_batches = int(len(train_set) / config.batch_size) + (1 if len(train_set) % config.batch_size != 0 else 0)

        for i in range(num_batches):
            end_index = min(config.batch_size * (i + 1), len(train_set))
            batch = train_set[i * config.batch_size:end_index]

            if len(batch) == 0:
                continue

            input_id_tensors = torch.stack([data[0] for data in batch])
            input_mask_tensors = torch.stack([data[1] for data in batch])
            label_tensors = torch.stack([data[2] for data in batch])

            b_input_ids = input_id_tensors.to(device)
            b_input_mask = input_mask_tensors.to(device)
            b_labels = label_tensors.long().to(device)

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs["loss"]
            total_train_loss += loss.item()

            # Accumulate gradients
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            # Perform optimizer step after accumulating gradients for gradient_accumulation_steps
            if (i + 1) % config.gradient_accumulation_steps == 0 or i == num_batches - 1:  # Ensure step is taken on the last batch
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        print(f"Total loss: {total_train_loss}")
        report = get_validation_performance(val_set, model, device, label_dict, config.batch_size)
        print(report)
        analyze_generalization(model, val_set, tokenizer, train_sentences)

    print("")
    print(f"Training complete at learning rate: {lr}!")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = extract_model_name(model_name)
    print(f"{model_name} model with lr {lr} saved at: {timestamp}")
    torch.save(model.state_dict(), f'models/{model_name}_{lr}_{timestamp}.pth')

print("")
print(f"Training complete!")