from src.config import max_length
import pandas as pd
import torch

def read_conll(file_path):
    """
    Reads a .conll file and returns a DataFrame with sentences and their corresponding labels.
    
    Parameters:
    file_path: Path to the .conll file.
    """
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                columns = line.split()
                word, label = columns[0], columns[-1]
                current_sentence.append(word)
                current_labels.append(label)
                
                # Check if the current word is a sentence boundary
                if word == '.' and label == 'O':
                    sentences.append(' '.join(current_sentence))
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []

    # Create a DataFrame from the accumulated sentences and labels
    df = pd.DataFrame({
        'sentences': sentences,
        'labels': labels
    })
    return df

def tokenize_and_format(sentences, tokenizer, max_length=max_length):
    """
    Tokenizes sentences and returns formatted input IDs and attention masks.
    
    Parameters:
    sentences: List of sentence strings to be tokenized.
    tokenizer: Tokenizer instance used for tokenizing the sentences.
    """
    input_ids = []
    attention_masks = []

    # Encode each sentence
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Adjust based on your model's maximum input length
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Store the input ID and the attention mask of this sentence
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert lists of tensors to single tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def get_dataset(df, tokenizer, label_dict, max_length=max_length):
    """
    Processes a DataFrame to return a dataset suitable for training/testing an NER model.
    
    Parameters:
    df: DataFrame containing 'Tokens' and 'Labels' columns.
    tokenizer: Tokenizer to use for encoding the sentences.
    label_dict: Dictionary mapping label names to indices.
    max_length: Maximum length of the tokenized input.
    """
    sentences = df.sentences.values
    
    # Tokenize sentences
    input_ids, attention_masks = tokenize_and_format(sentences, tokenizer, max_length)

    # Prepare labels
    label_list = []
    for labels in df.labels.values:
        # Initialize a list to hold the encoded labels for each sentence
        encoded_labels = [label_dict[label] for label in labels]
        
        # Truncate or pad the labels to match the max_length
        encoded_labels = encoded_labels[:max_length]  # Truncate if needed
        encoded_labels += [label_dict['O']] * (max_length - len(encoded_labels))  # Pad with 'O' if needed
        
        label_list.append(encoded_labels)

    # Convert label_list to a tensor
    labels = torch.tensor(label_list, dtype=torch.long)

    # Create the dataset
    dataset = [(input_ids[i], attention_masks[i], labels[i]) for i in range(len(df))]

    return dataset, sentences

def preprocess_entities(labels, tokens):
    """
    Extract entities from token-label pairs.
    
    Args:
    labels (list of int): List of label indices corresponding to each token.
    tokens (list of str): List of tokens corresponding to each label index.
    
    Returns:
    list of tuples: Each tuple represents an entity with (entity_type, start_index, end_index, entity_text).
    """
    entities = []
    current_entity = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label.split("-")[1]
            current_entity = (entity_type, i, i, token)
        elif label.startswith("I-") and current_entity and label.split("-")[1] == current_entity[0]:
            current_entity = (current_entity[0], current_entity[1], i, current_entity[3] + " " + token)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities