# Twitter Sentiment Extraction & Classification

This project focuses on extracting the sentiment-bearing span from a tweet and simultaneously classifying the overall sentiment (positive, negative, or neutral). 
It was built using a RoBERTa-based deep learning model and fine-tuned on the Twitter Sentiment Extraction dataset originally featured in a Kaggle competition.

## Project Goal

To build an end-to-end model that:

1. Identifies the relevant subspan of a tweet that reflects its sentiment.
   
2. Predicts the sentiment label (positive, neutral, negative) of the tweet.

3. Achieves a high Jaccard similarity score between the predicted and true spans.
 
## This project was created for resume demonstration and self-learning purposes, with a focus on mastering:

- Multitask learning
- Transformers (BERT → RoBERTa)
- Model optimization and loss balancing
- Deployment-ready architecture

## Dataset Used

Source : [Kaggle](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data)


## Model Architecture

###Base

- Switched from BERT to RoBERTa-base for better performance on short text.

### Custom Head

- _span_fc_  : Linear layer for span prediction (start and end token logits).

- _classifier_ : Linear layer for 3-class sentiment prediction.

### Outputs

- _start_logits_ , _end_logits_ → _extract_ _span_

- _sentiment_logits_ → _classify_ _sentiment_

## Key Features and Decisions

- **Custom Weighted Loss**: Heavier weight (0.7) on start token prediction.

- **Early Stopping**: Patience of 2 epochs.

- **Learning Rate Tuning** : Final learning rate of 1e-5 gave best results.

- Model Saving: Torch model saved using ``` torch.save() ```

- **Multitask Learning**: Jointly trained for span extraction + sentiment classification.

## Experiments + Trial & Error

| Attempt | Changes Made | Result |
|--------|--------------|--------|
| #1 | Basic BERT model | 0.83 Jaccard but no sentiment classifier |
| #2 | Weighted loss, adjusted epochs | Drop in performance, overfitting observed |
| #3 | Switched to RoBERTa, multitask output | Best generalization, stable results |
| #4 | Lowered LR to 1e-5 | Reduced loss significantly |


## Final Results

- **Best Jaccard Score**: 0.8392

- **Validation Loss**: ≈ 0.85

- **Visual Inspection**: Predicted spans were accurate and contextually meaningful.

## How To Run
```
git clone https://github.com/pbattu18/twt_sent
cd twitter-sentiment-model
pip install -r requirements.txt
```
Use Kaggle, Jupyter Notebooks to run ```twitter_sent_model.ipynb```


## Future Work

- Build a Streamlit UI or Kaggle interactive demo

- Improve span prediction with pointer networks

- Test generalization on other Twitter datasets

- Extend to emotion classification
