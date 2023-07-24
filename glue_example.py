# !pip install datasets
# !pip install transformers[torch]
# !pip install accelerate -U

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# Loading the MRPC dataset
dataset = load_dataset("glue", "mrpc")
metric = load_metric('glue', 'mrpc')

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

# Evaluating the model without any finetuning
trainer.evaluate()
# {'eval_loss': 1.148370385169983,
#  'eval_accuracy': 0.3137254901960784,
#  'eval_f1': 0.0,
#  'eval_runtime': 127.4827,
#  'eval_samples_per_second': 3.2,
#  'eval_steps_per_second': 0.055}

# # Training the model
trainer.train()

# Evaluating the model
trainer.evaluate()
# {'eval_loss': 0.47797587513923645,
#  'eval_accuracy': 0.8504901960784313,
#  'eval_f1': 0.8942807625649914,
#  'eval_runtime': 2.9529,
#  'eval_samples_per_second': 138.171,
#  'eval_steps_per_second': 2.371,
#  'epoch': 3.0}