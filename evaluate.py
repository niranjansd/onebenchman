"""File to run evaluations.

Usage:
    python evaluate.py --train --dataset glue --task mrpc
"""
# Path: evaluate.py
# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, 
                    help="Whether to finetune the model on the task or not",
                    action="store_true")
parser.add_argument("--dataset", default='glue',
                    help="Dataset to use for training and evaluation")
parser.add_argument("--task", default='mrpc',
                    help="Task to use for training and evaluation")
args = parser.parse_args()

# print(args)
# print(args.train)
# print(args.dataset)
# print(args.task)

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset, load_metric


# Loading the MRPC dataset
dataset = load_dataset(args.dataset, args.task)
metric = load_metric(args.dataset, args.task)

# Loading BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenizing the dataset
def encode(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

dataset = dataset.map(encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Splitting the dataset
train_dataset = dataset['train']
eval_dataset = dataset['validation']

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

# Defining the Trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# # Training the model
if args.train:
    print('Training...')
    trainer.train()

# Evaluating the model
print("Eval")
trainer.evaluate()
