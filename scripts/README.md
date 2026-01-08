# Evaluation Scripts

## CNN/DailyMail Evaluation

### Install Dependencies

```bash
uv pip install datasets rouge-score
```

### Run Evaluation

Quick test (10 samples):

```bash
python eval_cnn_dailymail.py
```

### Full Evaluation

For complete evaluation, specify the split:

```bash
# Test set (11,490 samples)
python eval_cnn_dailymail.py --split test

# Validation set (11,332 samples)
python eval_cnn_dailymail.py --split validation
```

