# Import libraries
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import numpy as np
import os

# WAND tracking
#os.environ["WANDB_PROJECT"] = "ml-toxicity-classifier"
#os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Load the toxicity dataset
data = load_dataset("HU-Berlin-ML-Internal/toxicity-dataset")

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = data.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

# Evaluation metrics
metric = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(np.asarray(logits), axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Train the model
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    per_device_train_batch_size=16,
    logging_steps=100,
    learning_rate=3e-5,
    num_train_epochs=5,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# Save the model
trainer.model.save_pretrained("./models/toxicity-classifier")
tokenizer.save_pretrained("./models/toxicity-classifier")