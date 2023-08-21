# DuluthNLP at SemEval-2023 Task 12

This repository contains the source code for our paper [DuluthNLP at SemEval-2023 Task 12: AfriSenti-SemEval: Sentiment Analysis for Low-resource African Languages using Twitter Dataset](https://aclanthology.org/2023.semeval-1.236). The paper includes a description of a pretrained model, described below, that was trained from scratch on Twi, the predominant language in Ghana.

# TwiBERT
## Model Description
TwiBERT is a pre-trained language model specifically designed for the Twi language, which is widely spoken in Ghana, West Africa. This model has 61 million parameters, 6 layers, 6 attention heads, 768 hidden units, and a feed-forward size of 3072. To optimize its performance, TwiBERT was trained using a combination of the Asanti Twi Bible and a dataset sourced through crowdsourcing efforts.

## Limitations:
The model was trained on a relatively limited dataset (approximately 5MB), which may hinder its ability to learn intricate contextual embeddings and effectively generalize. Additionally, the dataset's focus on the Bible could potentially introduce a strong religious bias in the model's output.

## How to use it
You can use TwiBERT by finetuning it on a downtream task. The example code below illustrates how you can use the TwiBERT model on a downtream task:

``` python
from transformers import AutoTokenizer, AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained("sakrah/TwiBERT")
tokenizer = AutoTokenizer.from_pretrained("sakrah/TwiBERT")
```
