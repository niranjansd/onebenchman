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
def encode(task, tokenizer):
    feat = dataset_lib.GLUE['features'][task]
    def task_encode(examples):
        return tokenizer(*[examples[i] for i in feat],
                            truncation=True, padding='max_length', max_length=128)
    return task_encode

# Loading the model and tokenizer
def load_model_and_tokenizer(model_name):
    """Load a model and tokenizer."""
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer

# Evaluate all tasks in the GLUE benchmark
def evaluate_all_tasks(model_name, dataset_name):
    """Evaluate all tasks in the GLUE benchmark."""
    # Loading model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
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
    for task in dataset_lib.GLUE['features']:
        print(task)
        # Define datasets
        dataset = load_dataset(dataset_name, task)

        dataset = dataset.map(encode(task, tokenizer), batched=True)
        dataset.set_format(type='torch',
                           columns=['input_ids',
                                    'attention_mask',
                                    'label'])

        # Splitting the dataset
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

        # Defining the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metrics.compute_metrics_fn(task),
        )

        # # Training the model
        print('Training...')
        trainer.train()

        # Evaluating the model
        print("Eval")
        m[task] = trainer.evaluate()
        print(m[task])
    return m

args = parser.parse_args()
m = evaluate_all_tasks(model_name=args.model_name,
                       dataset_name=args.dataset_name)
mdf = pd.DataFrame(m)
mdf.to_csv(f'./results/{args.model_name}_{args.dataset_name}.csv')