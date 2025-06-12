import streamlit as st
import torch
from transformers import RobertaTokenizerFast
from tweet_model import TweetSentimentExtractor

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TweetSentimentExtractor()
model.load_state_dict(torch.load("tweet_sentiment_model.pt", map_location=device))
model.to(device)
model.eval()

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Prediction logic
def predict(text):
    sentiments = ['positive', 'neutral', 'negative']
    results = {}

    for sentiment in sentiments:
        inputs = tokenizer.encode_plus(
            sentiment, text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        offsets = inputs['offset_mapping'][0].tolist()

        with torch.no_grad():
            start_logits, end_logits, sentiment_logits = model(input_ids, attention_mask)

        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()

        if start_idx > end_idx:
            end_idx = start_idx

        selected_span = text[offsets[start_idx][0]: offsets[end_idx][1]]
        sentiment_score = torch.softmax(sentiment_logits, dim=1).cpu().numpy()[0]

        results[sentiment] = {
            "span": selected_span,
            "confidence": sentiment_score[sentiments.index(sentiment)]
        }

    # Pick best sentiment based on confidence
    best_sentiment = max(results.items(), key=lambda x: x[1]["confidence"])
    return best_sentiment[0], best_sentiment[1]["span"]

# Streamlit UI
st.title("Tweet Sentiment Extractor")
text = st.text_area("Enter a tweet:", "i love how supportive my friends are")

if st.button("Analyze"):
    sentiment, span = predict(text)
    st.markdown(f"**Predicted Sentiment:** `{sentiment}`")
    st.markdown(f"**Extracted Text:** _{span}_")
