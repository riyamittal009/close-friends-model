import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split 
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, create_optimizer

PT_TRANSFORMER = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(PT_TRANSFORMER)

model = TFAutoModelForSequenceClassification.from_pretrained(PT_TRANSFORMER)

# read in dataset
ds = load_dataset('csv', data_files='../data/ds1.csv')
# ds = pd.read_csv("~/sphere/close-friends-model/data/ds1.csv")

def tokenize(text):
    return tokenizer(text['Text'], truncation=True)

# tokenize dataset
tokenized_datasets = pd.DataFrame(ds.map(tokenize, batched=True))

new_df = pd.DataFrame(columns=['Text', 'attention_mask', 'input_ids', 'Label'])
# print(new_df.head())

for i in range(len(tokenized_datasets)):
    new_df.at[i,'Text'] = tokenized_datasets.iloc[i]['train']['Text']
    new_df.at[i, 'attention_mask'] = tokenized_datasets.iloc[i]['train']['attention_mask']
    new_df.at[i, 'input_ids'] = tokenized_datasets.iloc[i]['train']['input_ids']
    new_df.at[i, 'Label'] = tokenized_datasets.iloc[i]['train']['Label']

print(new_df.head())

# split dataset into train and test
train, test = train_test_split(new_df, test_size=0.2, random_state=42)

train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = train.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["Label"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = test.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["Label"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

# create optimizer
batch_size = 8
num_epochs = 3
batches_per_epoch = len(tokenized_datasets["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

# problem is here -- unable to fine tune Text Classification model with .fit()
model.fit(x=tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

