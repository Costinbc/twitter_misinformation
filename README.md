## Mitigation and Detection of Misinformation on X (formerly known as Twitter)

### Rumor-detection-acl-2017 - BERTweet-base

The model was fine-tuned on a merged dataset from Twitter15 and Twitter16 for 4 epochs. Below are the final evaluation metrics:

- **Dataset size**: ~2100 distinct-tweets
- **Labels**: True, False, Unverified, Non-rumor*
- **Validation Accuracy**: 81.1%
- **Macro F1 Score**: 0.798
- **Validation Loss**: 0.587
- **Evaluation Speed**: 1119 tweets/sec
- **Epochs**: 4

*currently unused 
