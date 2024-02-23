from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
processing_device = "cuda:0" if torch.cuda.is_available() else "cpu"

text_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(processing_device)
sentiment_labels = ["positive", "negative", "neutral"]

def analyze_news_sentiment(articles):
    if articles:
        encoded_articles = text_tokenizer(articles, return_tensors="pt", padding=True).to(processing_device)

        sentiment_logits = sentiment_model(encoded_articles["input_ids"], attention_mask=encoded_articles["attention_mask"])["logits"]
        sentiment_logits_sum = torch.nn.functional.softmax(torch.sum(sentiment_logits, 0), dim=-1)
        max_probability = sentiment_logits_sum[torch.argmax(sentiment_logits_sum)]
        determined_sentiment = sentiment_labels[torch.argmax(sentiment_logits_sum)]
        return max_probability, determined_sentiment
    else:
        return 0, sentiment_labels[-1]
