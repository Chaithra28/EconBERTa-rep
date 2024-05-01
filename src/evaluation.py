from sklearn.metrics import classification_report
from src.utils import reverse_label_dict
import numpy as np
import torch

# Function to get the validation performance
def get_validation_performance(val_set, model, device, label_dict, batch_size):
    # Put the model in evaluation mode
    model.eval()

    # Tracking variables
    total_eval_loss = 0
    all_pred_labels = []
    all_true_labels = []

    num_batches = int(len(val_set) / batch_size) + (1 if len(val_set) % batch_size != 0 else 0)

    for i in range(num_batches):
        end_index = min(batch_size * (i + 1), len(val_set))
        batch = val_set[i * batch_size:end_index]

        if len(batch) == 0:
            continue

        input_id_tensors = torch.stack([data[0] for data in batch])
        input_mask_tensors = torch.stack([data[1] for data in batch])
        label_tensors = torch.stack([data[2] for data in batch])

        # Move tensors to the GPU
        b_input_ids = input_id_tensors.to(device)
        b_input_mask = input_mask_tensors.to(device)
        b_labels = label_tensors.to(device)
        b_labels = b_labels.long()

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Get the predicted labels
            pred_labels = np.argmax(logits, axis=2).flatten()
            true_labels = label_ids.flatten()

            # Convert labels to their original names
            pred_labels = [reverse_label_dict.get(label) for label in pred_labels]
            true_labels = [reverse_label_dict.get(label) for label in true_labels]

            # Filter out special tokens ('O' label is used for non-entity and special tokens)
            filtered_pred_labels = [pred for pred, true in zip(pred_labels, true_labels) if true != 'O']
            filtered_true_labels = [true for true in true_labels if true != 'O']
            
            # After filtering out special tokens
            if not filtered_pred_labels or not filtered_true_labels:
                print("Warning: No non-'O' labels found in this batch.")
            else:
                all_pred_labels.extend(filtered_pred_labels)
                all_true_labels.extend(filtered_true_labels)
            
    # After processing all batches, check if we have any labels to report on
    if not all_true_labels or not all_pred_labels:
        print("Error: No non-'O' labels found in the entire validation set.")
        default_labels = [list(label_dict.values())[0]]  # Use the first label as a placeholder
        report = classification_report(default_labels, default_labels, digits=4, zero_division=0)
    else:
        # Calculate precision, recall, and F1 score
        report = classification_report(all_true_labels, all_pred_labels, digits=4, zero_division=0)

    return report
