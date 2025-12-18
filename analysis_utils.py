import pandas as pd

def get_detailed_stats(df):
    """Calculates basic statistics for the dashboard metrics."""
    stats = {
        "total_comments": len(df),
        "unique_commenters": df['author'].nunique(),
        "avg_words": df['text'].str.split().str.len().mean(),
        "avg_score": (df['sentiment_class'] - 1).mean()
    }
    return stats

def get_top_comments(df, sentiment_label, count=5):
    """
    Returns the top N comments for a specific sentiment.
    0 = Negative, 1 = Neutral, 2 = Positive
    """
    subset = df[df['sentiment_class'] == sentiment_label]
    # Sort by confidence to show the most 'certain' predictions first
    return subset.sort_values(by='confidence', ascending=False).head(count)