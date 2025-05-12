from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from huggingface_hub import login


def load_model(path:str, device:str, labels:dict) -> (AutoModelForSequenceClassification, AutoTokenizer):
    """
    load the model into the given device
    :param path: path to load the model from
    :param device: device to load the model to
    :param labels: data labels
    :return: tuple of (model, tokenizer)
    """
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=len(labels),
        id2label=labels, # todo I'm not entirely sure if these (id2label and label2id) are needed see #7
        label2id={v:k for k,v in labels.items()},
        ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        pretrained_model.resize_token_embeddings(len(tokenizer))
    return pretrained_model, tokenizer

def download_model(path:str, device:str, labels:dict, model_name:str, token:str) -> None:
    login(token)
    model, tokenizer = load_model(model_name, device, labels)

    if not os.path.exists(path):
        os.makedirs(path)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


# tokenize the dataset
def tokenize_function(examples, tokenizer, max_length):
    text = examples["text"]

    # Tokenize texts in batch mode
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    return encoding

