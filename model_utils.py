import emoji
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 115 

def contains_emoji(text):
    """Returns True if the string contains any emoji."""
    return emoji.emoji_count(text) > 0

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_predictions(texts, model, tokenizer):
    results = []
    confidences = []
    
    for t in texts:
        # IF EMOJI EXISTS: Skip prediction (assign a placeholder or Neutral)
        if contains_emoji(t):
            results.append(1) # Default to Neutral (1)
            confidences.append(0.0)
        else:
            cleaned = clean_text(t)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            pred = model.predict(padded)
            results.append(np.argmax(pred))
            confidences.append(np.max(pred))
            
    return np.array(results), np.array(confidences)