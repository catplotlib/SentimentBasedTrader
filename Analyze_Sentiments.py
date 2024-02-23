from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 

# Set the device to CUDA if available, otherwise use CPU
processing_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model for sentiment analysis from the FinBERT pre-trained model
text_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(processing_device)
sentiment_labels = ["positive", "negative", "neutral"]  # Define the sentiment labels

def analyze_news_sentiment(articles):
    # Analyzes the sentiment of a list of news articles.
    # If there are articles to analyze:
    if articles:
        # Tokenize the articles and prepare for model input
        encoded_articles = text_tokenizer(articles, return_tensors="pt", padding=True).to(processing_device)

        # Pass the tokenized articles through the model to get sentiment logits
        sentiment_logits = sentiment_model(encoded_articles["input_ids"], attention_mask=encoded_articles["attention_mask"])["logits"]
        # Sum the logits and apply softmax to get probabilities
        sentiment_logits_sum = torch.nn.functional.softmax(torch.sum(sentiment_logits, 0), dim=-1)
        # Find the maximum probability and its corresponding sentiment
        max_probability = sentiment_logits_sum[torch.argmax(sentiment_logits_sum)]
        determined_sentiment = sentiment_labels[torch.argmax(sentiment_logits_sum)]
        # Return the maximum probability and the determined sentiment
        return max_probability.item(), determined_sentiment
    else:
        # If no articles are provided, return 0 probability and neutral sentiment
        return 0, sentiment_labels[-1]
