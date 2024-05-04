# EconBERTa
Towards Robust Extraction of Named Entities in Economics

>This is a PyTorch Implementation

This repository contains our code for reproducing the paper [EconBERTa: Towards Robust Extraction of Named Entities in Economics](https://aclanthology.org/2023.findings-emnlp.774/) by Karim Lasri, Pedro Vitor Quinta de Castro, Mona Schirmer, Luis Eduardo San Martin, Linxi Wang, Tomáš Dulka, Haaya Naushan, John Pougué-Biyong, Arianna Legovini, and Samuel Fraiberger published at EMNLP Findings 2023.  

**This implementation is totally different from the author's implementation where he used allennlp for the most part to pretrain and finetune the models.** We, on the other hand, used Pytorch and Transformers primarily. You can find the author's implementation [here](https://github.com/worldbank/econberta-econie/tree/main).

This code demonstrates how to perform Named Entity Recognition (NER) using various transformer models and a Conditional Random Field (CRF) layer. The code is written in Python and utilizes the PyTorch and Transformers libraries.

**Authors**: [Ashutosh Pathak](https://www.linkedin.com/in/pathak-ash/), [Chaithra Bekal](https://www.linkedin.com/in/chaithra-bekal/), [Vikas Velagapudi](https://www.linkedin.com/in/vikas-velagapudi-48441a166/)

## Requirements

- Python 3.9
- PyTorch
- Transformers
- scikit-learn
- pandas
- pytorch-crf _(imported as torchcrf in the code)_

Make sure to install the required dependencies before running the code.

## Usage
> The main notebook is `notebook/EconBERTa.ipynb`
1. Install the required dependencies.
2. Place the dataset files (`train.conll`, `dev.conll`, `test.conll`) in the appropriate directory.
3. Set the desired `model_name` and other hyperparameters in the code.
4. Run the code cells in the provided order.
5. The training progress, validation performance, and test performance will be printed.
6. The trained model checkpoint will be saved for future use.

Feel free to experiment with different models and hyperparameters to achieve the best performance for your specific NER task.

## Models

The code supports the following transformer models:

- `worldbank/econberta` (default)
- `bert-base-uncased`
- `roberta-base`
- `mdeberta-v3-base`

To use a different model, simply replace the `model_name` variable with the desired model name and re-run the cells from that point onwards.

## Dataset

The code assumes the dataset is in CoNLL format and expects the following files:

- `train.conll`: Training data
- `dev.conll`: Validation data
- `test.conll`: Test data

Make sure to place these files in the appropriate directory before running the code.

## Preprocessing

The code performs the following preprocessing steps:

1. Reading the CoNLL files and converting them into pandas DataFrames.
2. Tokenizing the words using the specified transformer model's tokenizer.
3. Encoding the labels using a one-hot encoding scheme.
4. Creating PyTorch datasets by combining the tokenized input IDs, attention masks, and encoded labels.

The preprocessing steps ensure that the data is in the appropriate format for training and evaluation.

## Training

The code performs hyperparameter search by trying different learning rates specified in the `learning_rates` list. The training loop runs for a specified number of epochs (`max_epochs`) and uses the AdamW optimizer with a linear learning rate scheduler.

During training, the code evaluates the model on the validation set after each epoch and prints the classification report and entity-level metrics.

## Evaluation

After training, the code loads the best model checkpoint and evaluates it on the test set. It prints the classification report and entity-level metrics for the test set.

## Entity-level Metrics

The code calculates the following entity-level metrics:

- Exact Match (EM)
- Exact Boundary (EB)
- Partial Match (PM)
- Partial Boundaries (PB)
- Missed Label (ML)
- False Alarm (FA)

These metrics provide a detailed evaluation of the model's performance in recognizing named entities.

## Generalization Analysis

The code includes a function `analyze_generalization` that analyzes the model's generalization ability. It groups the entities based on their length and whether they were seen or unseen during training. It then calculates the entity-level metrics for each group and prints the results.

## Results

We were able to reproduce the trend in f1-scores among papers that the author talked about in the paper. We also implemented CheckList tests to assess model's robustness.

