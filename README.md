# Mitigation and Detection of Misinformation on X (formerly known as Twitter)

# Datasets
### Rumor-Detection-Acl 2017
A merged dataset from Twitter15 and Twitter16.

- **Dataset size**: 2000+ labeled tweets
- **Labels**: True, False, Unverified, Non-rumor*
  
*currently unused 

### MiDe22
A multi-event tweet dataset.

- **Dataset size**: 5000+ labeled tweets
- **Labels**: True, False, Other

### ANTiVax
A dataset of anti-vaccine tweets from November 2020 to July 2021.

- **Dataset size**: 5000+ labeled tweets
- **Labels**: True, False
  
## BERTweet-base model results

### Rumor-Detection-Acl 2017
- **Validation Accuracy**: 81.1%
- **Macro F1 Score**: 0.798
- **Validation Loss**: 0.587
- **Evaluation Speed**: 1670 tweets/sec
- **Epochs**: 4

### ANTiVax
- **Validation Accuracy**: 97.4%
- **Macro F1 Score**: 0.970
- **Validation Loss**: 0.122
- **Evaluation Speed**: 1829 tweets/sec
- **Epochs**: 4

### MiDe22
- **Validation Accuracy**: 80.2%
- **Macro F1 Score**: 0.753
- **Validation Loss**: 0.590
- **Evaluation Speed**: 1083 tweets/sec
- **Epochs**: 4

## RoBERTa-base fine-tuned for irony detection results

### Rumor-Detection-Acl 2017
- **Validation Accuracy**: 82.1%
- **Macro F1 Score**: 0.811
- **Validation Loss**: 0.879
- **Evaluation Speed**: 836 tweets/sec
- **Epochs**: 4

### ANTiVax
- **Validation Accuracy**: 97.5%
- **Macro F1 Score**: 0.972
- **Validation Loss**: 0.143
- **Evaluation Speed**: 1207 tweets/sec
- **Epochs**: 4

### MiDe22
- **Validation Accuracy**: 79.2%
- **Macro F1 Score**: 0.748
- **Validation Loss**: 0.549
- **Evaluation Speed**: 907 tweets/sec
- **Epochs**: 4

## XLnet base-size model results

### Rumor-Detection-Acl 2017
- **Validation Accuracy**: 73.4%
- **Macro F1 Score**: 0.730
- **Validation Loss**: 0.660
- **Evaluation Speed**: 930 tweets/sec
- **Epochs**: 6

### ANTiVax
- **Validation Accuracy**: 97.8%
- **Macro F1 Score**: 0.975
- **Validation Loss**: 0.117
- **Evaluation Speed**: 775 tweets/sec
- **Epochs**: 4

### MiDe22
- **Validation Accuracy**: 78.1%
- **Macro F1 Score**: 0.736
- **Validation Loss**: 0.756
- **Evaluation Speed**: 477 tweets/sec
- **Epochs**: 4
