import os
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from bert_dataset import BERTDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    set_seed,
)
from utils import check_git_status, create_experiment_dir, get_arguments, save_args

commit_id = check_git_status()
experiment_dir = create_experiment_dir(experiment_type="train")
model_args, data_args, training_args, json_args = get_arguments(experiment_dir)

filename = data_args.dataset_name.split(".")[-2].split("/")[-1]
set_seed(seed=training_args.seed)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path, num_labels=7
).to(DEVICE)

data = pd.read_csv(data_args.dataset_name)
dataset_train, dataset_valid = train_test_split(
    data, test_size=training_args.test_size, random_state=training_args.seed
)

data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if training_args.do_train:
    trainer.train()

if training_args.do_eval:
    dataset_test = pd.read_csv(
        os.path.join(training_args.output_dir, f"{filename}'.csv'")
    )
    model.eval()

if training_args.do_predict:
    preds = []

    for idx, sample in tqdm(
        dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"
    ):
        inputs = tokenizer(sample["text"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    dataset_test["target"] = preds
    dataset_test.to_csv(
        os.path.join(training_args.output_dir, f"{filename}_output.csv"), index=False
    )

if training_args.do_train or training_args.do_eval or training_args.do_predict:
    save_args(json_args, experiment_dir, commit_id)
