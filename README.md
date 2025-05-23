# Mitigation and Detection of Misinformation on X (formerly known as Twitter)

## BERTweet-base

### Rumor-Detection-Acl 2017
The model was fine-tuned on a merged dataset from Twitter15 and Twitter16. Below are the final evaluation metrics:

- **Dataset size**: 2000+ labeled tweets
- **Labels**: True, False, Unverified, Non-rumor*
- **Validation Accuracy**: 81.1%
- **Macro F1 Score**: 0.798
- **Validation Loss**: 0.587
- **Evaluation Speed**: 1670 tweets/sec
- **Epochs**: 4

*currently unused 

### ANTiVax
The model continued training on a dataset of anti-vaccine tweets from November 2020 to July 2021. Below are the final evaluation metrics:

- **Dataset size**: 5000+ labeled tweets
- **Labels**: True, False
- **Validation Accuracy**: 97.4%
- **Macro F1 Score**: 0.970
- **Validation Loss**: 0.122
- **Evaluation Speed**: 1829 tweets/sec
- **Epochs**: 4

### MiDe22
The model continued training on a multi-event tweet dataset. Below are the final evaluation metrics:

- **Dataset size**: 5000+ labeled tweets
- **Labels**: True, False, Other
- **Validation Accuracy**: 80.2%
- **Macro F1 Score**: 0.753
- **Validation Loss**: 0.590
- **Evaluation Speed**: 1083 tweets/sec
- **Epochs**: 4
