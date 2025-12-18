import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import re
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
from collections import Counter

# Import your custom modules
from model_utils import get_predictions
from youtube_utils import fetch_data
from analysis_utils import get_detailed_stats, get_top_comments
from summary_utils import generate_summary

# ---------------- CONFIG ----------------
st.set_page_config(page_title="YouTube Intelligence", layout="wide")

# ---------------- CACHE MODELS ----------------
@st.cache_resource
def load_assets():
    model = load_model("sentiment_model.h5", compile=False)
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_assets()

def extract_video_id(url):
    pattern = r"(?:v=|\/|be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# ---------------- SESSION STATE ----------------
for key in ["transcript", "df", "stats", "current_summary"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- HEADER ----------------
st.title(" YouTube Intelligence Dashboard")
st.caption("Advanced AI Sentiment Analysis & Video Insights")
st.divider()

# ---------------- INPUT SECTION ----------------
col_in, col_pre = st.columns([1, 1], gap="large")
with col_in:
    st.subheader("Video Analysis")
    video_url = st.text_input("YouTube URL", placeholder="Paste link here...")
    analyze_btn = st.button("Run Deep Analysis", use_container_width=True)

with col_pre:
    v_id = extract_video_id(video_url)
    if v_id:
        st.image(f"https://img.youtube.com/vi/{v_id}/maxresdefault.jpg", use_container_width=True)
    else:
        st.info("Paste a link to preview video.")

# ---------------- ANALYSIS LOGIC ----------------
if analyze_btn and video_url:
    with st.spinner(" Extracting Data..."):
        transcript, comments = fetch_data(video_url)

    if not comments:
        st.error("No comments found.")
    else:
        df_raw = pd.DataFrame(comments)
        classes, confs = get_predictions(df_raw["text"], model, tokenizer)
        df_raw["sentiment_class"] = classes
        df_raw["confidence"] = confs
        df_raw["label"] = df_raw["sentiment_class"].map({0: "Negative", 1: "Neutral", 2: "Positive"})
        
        st.session_state["transcript"] = transcript
        st.session_state["df"] = df_raw
        st.session_state["stats"] = get_detailed_stats(df_raw)
        st.session_state["current_summary"] = None
        st.rerun()

#  DASHBOARD DISPLAY 
if st.session_state.get("df") is not None:
    df = st.session_state["df"]
    stats = st.session_state["stats"]

    #  ROW 1: METRICS & GAUGE 
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        st.metric("Total Comments", f"{stats['total_comments']:,}")
        st.metric("Unique Users", f"{stats['unique_commenters']:,}")
    with c2:
        st.metric("Avg Words", f"{stats['avg_words']:.1f}")
        vibe = "Positive" if stats["avg_score"] > 0.1 else "Negative" if stats["avg_score"] < -0.1 else "Neutral"
        st.metric("Overall Vibe", vibe)

    with c3:
        # NEW: Sentiment Gauge Chart
        score = stats["avg_score"] # Range -1 to 1
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Sentiment Intensity (-1 to +1)"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [-1, -0.2], 'color': "#EF553B"},
                    {'range': [-0.2, 0.2], 'color': "#636EFA"},
                    {'range': [0.2, 1], 'color': "#00CC96"}]}))
        fig_gauge.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ---- ROW 2: AI SUMMARY ----
    with st.expander("AI Video Summary", expanded=True):
        if st.session_state["transcript"]:
            if st.button("✨ Generate Summary"):
                st.session_state["current_summary"] = generate_summary(st.session_state["transcript"])
            if st.session_state["current_summary"]:
                st.info(st.session_state["current_summary"])
        else:
            st.warning("No transcript available.")

    # ROW 3: GRAPHS 
    st.divider()
    g1, g2 = st.columns(2)
    
    with g1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(df, names="label", color="label", hole=0.4,
                         color_discrete_map={"Positive": "#00CC96", "Neutral": "#636EFA", "Negative": "#EF553B"})
        st.plotly_chart(fig_pie, use_container_width=True)

    with g2:
        st.subheader("Top Mentioned Keywords")
        words = " ".join(df["text"]).lower().split()
        # Filter: keep words > 3 chars to avoid 'the', 'is'
        filtered_words = [w for w in words if len(w) > 3]
        word_counts = Counter(filtered_words).most_common(10)
        word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        fig_bar = px.bar(word_df, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- ROW 4: WORDCLOUD & TABS ----
    st.divider()
    col_wc, col_tabs = st.columns([1, 1])

    with col_wc:
        st.subheader("Keyword Cloud")
        wc = WordCloud(width=600, height=400, background_color="white").generate(" ".join(df["text"]))
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)

    with col_tabs:
        st.subheader("Comment Explorer")
        t1, t2, t3 = st.tabs(["Positive", "Neutral", "Negative"])
        with t1:
            for _, r in get_top_comments(df, 2).iterrows():
                st.caption(f"**{r['author']}**")
                st.write(r['text'])
        with t2:
            for _, r in get_top_comments(df, 1).iterrows():
                st.caption(f"**{r['author']}**")
                st.write(r['text'])
        with t3:
            for _, r in get_top_comments(df, 0).iterrows():
                st.caption(f"**{r['author']}**")
                st.write(r['text'])