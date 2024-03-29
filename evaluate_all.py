"""File to run evaluations.

Usage:
    python evaluate_all.py --dataset glue
"""
# Path: evaluate.py

# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert-base-uncased",
                    help="Dataset to use for training and evaluation")
parser.add_argument("--train", default=False, 
                    help="Whether to finetune the model on the task or not",
                    action="store_true")
parser.add_argument("--dataset_name", default='glue',
                    help="Dataset to use for training and evaluation")

# Import libraries
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import dataset_lib
import functools
import metrics
import pandas as pd

# Tokenizing the dataset
def encode(feat, tokenizer):
    def task_encode(examples):
        return tokenizer(*[examples[i] for i in feat],
                            truncation=True, padding='max_length', max_length=128)
    return task_encode

# Loading the model and tokenizer
def load_model_and_tokenizer(model_name, classifier=True):
    """Load a model and tokenizer."""
    if classifier:
        model = BertForSequenceClassification.from_pretrained(model_name)
    else:
        model = BertForSequenceClassification.from_pretrained(model_name,
                                                              num_labels=1,
                                                              ignore_mismatched_sizes=True)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer

# Evaluate all tasks in the GLUE benchmark
def evaluate_all_tasks(model_name, dataset_name, train=False):
    """Evaluate all tasks in the GLUE benchmark."""
    # Loading model and tokenizer
    datadict = dataset_lib.DATASET[dataset_name]
    
    # Defining training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    m = {}
    for task in datadict:
        model, tokenizer = load_model_and_tokenizer(model_name, classifier=task != 'stsb')
        print(task)
        # Define datasets
        if dataset_name == 'glue':
            dataset = load_dataset(dataset_name, task.split('(')[0])
        else:
            dataset = load_dataset(task)

        dataset = dataset.map(encode(datadict[task]['features'], tokenizer), batched=True)
        dataset.set_format(type='torch',
                           columns=['input_ids',
                                    'attention_mask',
                                    'label'])

        # Splitting the dataset
        train_dataset = dataset['train']
        eval_dataset = dataset[datadict[task]['eval_split']]

        # Defining the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metrics.compute_metrics_fn(task),
        )

        # # Training the model
        if train:
            print('Training...')
            trainer.train()

        # Evaluating the model
        print("Eval")
        m[task] = trainer.evaluate()
        print(m[task])
    return m

args = parser.parse_args()
m = evaluate_all_tasks(model_name=args.model_name,
                       dataset_name=args.dataset_name,
                       train=args.train)
mdf = pd.DataFrame(m)
mdf.to_csv(f'./results/{args.model_name}_{args.dataset_name}.csv')