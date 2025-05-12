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
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import os
import argparse

from helpers import model as helpers

DEVICE = "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B" 
DATA_PATH = "data"
RANDOM_SEED = 42
LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 1
OUTPUT_DIR = "output"
MAX_LENGTH = 5000

# label maps
id2label = {0: "Normal", 1: "Suspicious"}
label2id = {v:k for k,v in id2label.items()}

# load the dataset
def load_output_dataset(path, random_seed):
    dataset = load_dataset("json", data_files=path)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=random_seed)
    print("Dataset loaded.")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    print(f"Dataset structure: {dataset}")
    return dataset

# load the config.yaml file
def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# set the config values
def set_config(config):
    global MODEL_NAME, DATA_PATH, RANDOM_SEED, LR, BATCH_SIZE, EPOCHS, OUTPUT_DIR, MAX_LENGTH, DEVICE
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
    if config["max_length"]:
        MAX_LENGTH = int(config["max_length"])
    if config["device"]:
        DEVICE = config["device"]
    print("Config loaded.")

# clear the cuda cache
def clear_cache():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()


def run():
    """
    sets up and runs the trainer
    """

    # clear cache
    clear_cache()
    # load pretrained model
    path = os.path.join(MODEL_NAME)
    print("Loading pretrained model...")
    try:
        pre_trained_model, tokenizer = helpers.load_model(path, DEVICE, label2id)
    except Exception as e:
        print("Error loading pretrained model.")
        print(e)
        exit(1)
    # load the dataset
    print("Loading dataset...")
    dataset = load_output_dataset(DATA_PATH, RANDOM_SEED)
    # data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: helpers.tokenize_function(x, tokenizer, MAX_LENGTH), batched=True)
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
        output_dir=MODEL_NAME + "-lora-text-classification",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
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
    # save the model
    print("Saving the model...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    trainer.save_state()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved.")


if __name__ == "__main__":
    # read the config.yaml file
    config = load_config()
    # set the config values 
    set_config(config)

    # * setup argparse
    parser = argparse.ArgumentParser(
        prog='classifier',
        description='basic training script for classifying siem events'
    )
    parser.add_argument('-i', '--input',
                        help='path to input data',
                        type=str,
                        default=DATA_PATH)
    parser.add_argument('-o', '--output',
                        help='path to output model',
                        type=str,
                        default=OUTPUT_DIR)
    # override defaults
    args = parser.parse_args()

    DATA_PATH = args.input
    OUTPUT_DIR = args.output

    run()