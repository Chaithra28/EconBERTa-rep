import random
import torch
import numpy as np    
from enum import Enum
from collections import defaultdict
from src.data_preprocessing import preprocess_entities

# Define the label dictionary
label_dict = {
    'O': 0,
    'B-intervention': 1,
    'I-intervention': 2,
    'B-outcome': 3,
    'I-outcome': 4,
    'B-population': 5,
    'I-population': 6,
    'B-effect_size': 7,
    'I-effect_size': 8,
    'B-coreference': 9,
    'I-coreference': 10
}

# Create the reverse label dictionary
reverse_label_dict = {v: k for k, v in label_dict.items()}

# Set the device
device = torch.device("cuda")

# Ensure reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_entity_level_metrics(true_entities, pred_entities):
    """
    Compute entity-level metrics for a single document.
    
    Parameters:
    true_entities (list): List of true entities in the document.
    pred_entities (list): List of predicted entities in the document.
    
    Returns:
    dict: A dictionary containing the entity-level metrics.
    """
    metrics = {"EM": 0, "EB": 0, "PM": 0, "PB": 0, "ML": 0, "FA": 0}
    true_matched = set()
    pred_matched = set()

    # Check for exact and partial matches
    for i, true_entity in enumerate(true_entities):
        for j, pred_entity in enumerate(pred_entities):
            if j in pred_matched:
                continue
            if true_entity == pred_entity:
                metrics["EM"] += 1
                true_matched.add(i)
                pred_matched.add(j)
                break
            elif true_entity[0] == pred_entity[0] and is_overlapping((true_entity[1], true_entity[2]), (pred_entity[1], pred_entity[2])):
                if true_entity[1] == pred_entity[1] and true_entity[2] == pred_entity[2]:
                    metrics["EB"] += 1
                else:
                    metrics["PM"] += 1
                true_matched.add(i)
                pred_matched.add(j)
                break
            elif is_overlapping((true_entity[1], true_entity[2]), (pred_entity[1], pred_entity[2])):
                metrics["PB"] += 1
                true_matched.add(i)
                pred_matched.add(j)
                break

    # Check for missed labels (entities in true but not in pred)
    for i, true_entity in enumerate(true_entities):
        if i not in true_matched:
            metrics["ML"] += 1

    # Check for false alarms (entities in pred but not in true)
    for j, pred_entity in enumerate(pred_entities):
        if j not in pred_matched:
            metrics["FA"] += 1

    return metrics

def is_overlapping(span1, span2):
    """
    Check if two spans overlap.
    Args:
    span1, span2 (tuple): (start_index, end_index) of the span.

    Returns:
    bool: True if spans overlap, False otherwise.
    """
    assert len(span1) == 2 and len(span2) == 2, "Each span must be a tuple of two elements (start_index, end_index)"
    start1, end1 = span1
    start2, end2 = span2
    return max(start1, start2) <= min(end1, end2)

def analyze_generalization(model, data, tokenizer, train_words):
    """
    Analyze the generalization of the model on the given data.
    
    Parameters:
    model (torch.nn.Module): The trained model.
    data (list): List of tuples containing input_ids, attention_mask, and label_tensor.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to convert tokens to input_ids.
    train_words (set): Set of words seen in the training data.
    
    Returns:
    (list, dict): List of group names and a dictionary containing the entity-level metrics of each group.
    """
    grouped_entities = defaultdict(lambda: ([], []))  # {group_name: (true_entities, pred_entities)}        
    groups=[]
    mtrcs=[]

    for i, (input_ids, attention_mask, label_tensor) in enumerate(data):
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        
        # Call model without labels to get the decoded labels
        with torch.no_grad():
            decoded_labels = model(input_ids, attention_mask=attention_mask)["decoded"][0]
            # No need to use argmax since CRF.decode returns the most likely tag sequence
        
        # Convert the decoded labels to label names using label_dict
        pred_labels = [reverse_label_dict.get(label) for label in decoded_labels]

        # Convert input_ids to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist(), skip_special_tokens=True)

        # Assuming true_labels are provided in a similar structure
        true_labels = [reverse_label_dict.get(l.item()) for l in label_tensor]
        
        # Preprocess entities for true and predicted labels
        true_entities = preprocess_entities(true_labels, tokens)
        pred_entities = preprocess_entities(pred_labels, tokens)
        
        for true_entity, pred_entity in zip(true_entities, pred_entities):
            length = true_entity[2] - true_entity[1]

            seen = any(word in train_words for word in true_entity[3].split())  # Check if any word in entity text was seen in training

            group_name = f"Length {length} - {'Seen' if seen else 'Unseen'}"
            grouped_entities[group_name][0].append(true_entity)
            grouped_entities[group_name][1].append(pred_entity)
    
    for group_name, group_data in grouped_entities.items():
        group_true_entities, group_pred_entities = group_data
        metrics = compute_entity_level_metrics(group_true_entities, group_pred_entities)
        print(f"Group: {group_name}, Metrics: {metrics}")
        groups.append(group_name)
        mtrcs.append(metrics)
        
    return groups, mtrcs

def extract_model_name(full_string):
    # Split the string by the '/' character
    parts = full_string.split('/')
    if len(parts) == 2:
        return parts[1]  # The second part is the model name
    else:
        raise ValueError("Invalid model_name format. Expected 'org_name/model_name'.")  # Invalid format
    

# Define an enum for model names
class ModelName(Enum):
    BERT = 'google-bert/bert-base-uncased'
    BERT_multilingual = 'google-bert/bert-base-multilingual-uncased'
    ROBERTA = 'FacebookAI/roberta-base'
    XLM_ROBERTA = 'FacebookAI/xlm-roberta-base'
    MDEBERTA = 'microsoft/mdeberta-v3-base'
    ECONBERTA_FC = 'worldbank/econberta'
    ECONBERTA_FS = 'worldbank/econberta-fs'
    
