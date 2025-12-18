Advanced Sentiment Analytics and Video Summarization
The YouTube Intelligence Dashboard is a data science application designed to bridge the gap between video content and viewer perception. By combining deep learning for sentiment classification with large language models for summarization, this tool provides a comprehensive overview of how an audience is reacting to any given video.
Overview
Modern content creators and marketers face the challenge of processing thousands of comments to understand audience sentiment. This platform automates that process by fetching up to 1000 comments, analyzing them through a custom-trained Long Short-Term Memory (LSTM) network, and generating concise notes from the video transcript using the Mistral-7B-Instruct model.
The system is architected to prioritize data integrity. It handles the nuances of modern communication by filtering emoji-heavy comments from the AI prediction pipeline to maintain high accuracy, while ensuring the user interface remains intuitive and visually informative.
Technical Architecture
The application is built on a modular Python framework, utilizing several specialized components to manage the data lifecycle:
Data Acquisition Layer: Uses the YouTube Data API v3 with pagination logic to retrieve a significant sample size of user engagement. It simultaneously accesses the YouTube Transcript API to gather textual data for summarization.
Sentiment Analysis Engine: A TensorFlow-based LSTM model processes cleaned text. It is supported by a Scikit-learn tokenizer to convert natural language into numerical tensors.
Language Processing Layer: Long transcripts are managed through a Recursive Character Text Splitter. These segments are processed via the Hugging Face Inference API using Mistral-7B to synthesize a cohesive 250-word summary.
Presentation Layer: Built with Streamlit, the interface provides interactive Plotly visualizations, including sentiment distribution donut charts and intensity gauges.
Installation and Configuration
Environment Requirements
The project requires Python 3.8 or higher. It is recommended to use a virtual environment to manage dependencies.

Clone the repository:
git clone https://github.com/yourusername/youtube-intelligence.git
cd youtube-intelligence
pip install -r requirements.txt
Authentication: The application requires a .env file in the root directory to store sensitive credentials.

Code snippet

YOUTUBE_API_KEY=your_google_api_key_here
HF_TOKEN=your_hugging_face_token_here

Here is a professional, humanized version of your project documentation. It focuses on clarity, technical structure, and readability without the use of emojis.

YouTube Intelligence Dashboard
Advanced Sentiment Analytics and Video Summarization
The YouTube Intelligence Dashboard is a data science application designed to bridge the gap between video content and viewer perception. By combining deep learning for sentiment classification with large language models for summarization, this tool provides a comprehensive overview of how an audience is reacting to any given video.

Overview
Modern content creators and marketers face the challenge of processing thousands of comments to understand audience sentiment. This platform automates that process by fetching up to 1000 comments, analyzing them through a custom-trained Long Short-Term Memory (LSTM) network, and generating concise notes from the video transcript using the Mistral-7B-Instruct model.

The system is architected to prioritize data integrity. It handles the nuances of modern communication by filtering emoji-heavy comments from the AI prediction pipeline to maintain high accuracy, while ensuring the user interface remains intuitive and visually informative.

Technical Architecture
The application is built on a modular Python framework, utilizing several specialized components to manage the data lifecycle:

Data Acquisition Layer: Uses the YouTube Data API v3 with pagination logic to retrieve a significant sample size of user engagement. It simultaneously accesses the YouTube Transcript API to gather textual data for summarization.

Sentiment Analysis Engine: A TensorFlow-based LSTM model processes cleaned text. It is supported by a Scikit-learn tokenizer to convert natural language into numerical tensors.

Language Processing Layer: Long transcripts are managed through a Recursive Character Text Splitter. These segments are processed via the Hugging Face Inference API using Mistral-7B to synthesize a cohesive 250-word summary.

Presentation Layer: Built with Streamlit, the interface provides interactive Plotly visualizations, including sentiment distribution donut charts and intensity gauges.

Installation and Configuration
Environment Requirements
The project requires Python 3.8 or higher. It is recommended to use a virtual environment to manage dependencies.

Clone the repository:

Bash

git clone https://github.com/yourusername/youtube-intelligence.git
cd youtube-intelligence
Install dependencies:

Bash

pip install -r requirements.txt
Authentication: The application requires a .env file in the root directory to store sensitive credentials.

Code snippet

YOUTUBE_API_KEY=your_google_api_key_here
HF_TOKEN=your_hugging_face_token_here
Key Features
Intelligence Metrics
The dashboard displays a Sentiment Intensity Gauge that quantifies the vibe of the comment section on a scale from -1.0 to +1.0. This allows for an immediate understanding of whether a video is being received positively or facing significant backlash.
Keyword Extraction
Beyond simple sentiment, the system employs a frequency counter to identify the top 10 most mentioned words. This provides context on exactly what the audience is discussing, displayed alongside a WordCloud for visual density mapping.
Categorized Exploration
Users can explore the raw data through filtered tabs. These tabs organize comments into Positive, Neutral, and Negative categories, allowing for qualitative review of specific viewer feedback.
