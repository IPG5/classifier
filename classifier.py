import yaml
import json
from pprint import pprint
import evaluate
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    LlamaModel,
    AutoConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path
import random

DEVICE = "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B" 
DATA_PATH = "data"
RANDOM_SEED = 42
LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 1
OUTPUT_DIR = "output"

# label maps
id2label = {0: "Normal", 1: "Suspicious"}
label2id = {v:k for k,v in id2label.items()}

#load_dotenv()
#torch.cuda.empty_cache()
#torch.backends.cudnn.benchmark = True
#torch.cuda.reset_peak_memory_stats()

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def set_config(config):
    global MODEL_NAME, DATA_PATH, MODEL_DIR, TOKENIZER_DIR, RANDOM_SEED, LR, BATCH_SIZE, EPOCHS, OUTPUT_DIR
    if config["model_name"]:
        MODEL_NAME = config["model_name"]
    if config["data_path"]:
        DATA_PATH = config["data_path"]
    if config["random_seed"]:
        RANDOM_SEED = int(config["random_seed"])
    if config["learning_rate"]:
        LR = float(config["learning_rate"])
    if config["batch_size"]:
        BATCH_SIZE = int(config["batch_size"])
    if config["epochs"]:
        EPOCHS = int(config["epochs"])
    if config["output_dir"]:
        OUTPUT_DIR = config["output_dir"]
    print("Config loaded.")

def download_pretrained_model(path):
    load_dotenv()
    login(token=os.getenv("hugging_face_PAG"))
    model = AutoModelForSequenceClassification.from_pretrained(
                                                                MODEL_NAME, 
                                                                num_labels=len(id2label),
                                                                id2label=id2label,
                                                                label2id=label2id
                                                                ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))\
      
    if not os.path.exists(path):
        os.makedirs(path)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model, tokenizer

def load_pretrained_model(path):
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                                                                path,
                                                                num_labels=len(id2label),
                                                                id2label=id2label,
                                                                label2id=label2id
                                                                ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        pretrained_model.resize_token_embeddings(len(tokenizer))
    return pretrained_model, tokenizer

def tokenize_function(examples, tokenizer):
    text = examples["text"]

    # Tokenize texts in batch mode
    encoding = tokenizer(
        text,
        truncation=True, 
        padding="max_length", 
        max_length=5000,
        return_tensors="pt"
    )

    return encoding

def load_output_dataset(path):
    dataset = load_dataset("json", data_files=path)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=RANDOM_SEED)
    print("Dataset loaded.")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    print(f"Dataset structure: {dataset}")
    return dataset

def clear_cache():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    pre_trained_model, tokenizer = None, None
    # read the config.yaml file
    config = load_config()
    # set the config values 
    set_config(config)

    # clear cache
    clear_cache()

    # load or download the pretrained model
    pretrained_exist = config["pretrained_model_exists"]
    path = os.path.join(MODEL_NAME)
    if pretrained_exist == False:
        print("Downloading pretrained model...")
        pre_trained_model, tokenizer = download_pretrained_model(path)
    else:
        print("Loading pretrained model...")
        pre_trained_model, tokenizer = load_pretrained_model(path)
    
    # load the dataset
    print("Loading dataset...")
    dataset = load_output_dataset(DATA_PATH)

    # data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    print(f"Tokenized dataset structure: {tokenized_dataset}")

    # training arguments
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        r=4
    )
    print("PEFT config loaded.")
    print(peft_config)

    model = get_peft_model(pre_trained_model, peft_config)
    model.print_trainable_parameters()
    print("Model loaded.")

    # Explicitly set padding token in the model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # define training arguments
    training_args = TrainingArguments(
        output_dir= MODEL_NAME + "-lora-text-classification",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy = "epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        seed=RANDOM_SEED,
        label_names=["label"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Trainer created.")

    # train the model
    print("Training the model...")
    trainer.train()
    print("Model trained.")

    #save the model
    print("Saving the model...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    trainer.save_state()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved.")