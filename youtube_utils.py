import re
import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def extract_video_id(url):
    """Extracts the 11-character video ID from various YouTube URL formats."""
    pattern = r"(?:v=|\/|be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_data(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid URL", None
    
    # 1. Improved Transcript Fetching
    transcript = ""
    try:
        # Tries English first, then falls back to other languages if available
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # You can prioritize languages here, e.g., ['en', 'es', 'fr']
        ts = transcript_list.find_transcript(['en']).fetch()
        transcript = " ".join([i['text'] for i in ts])
    except (TranscriptsDisabled, NoTranscriptFound):
        transcript = "Transcript is disabled or not found for this video."
    except Exception as e:
        transcript = f"An error occurred fetching transcript: {str(e)}"

    # 2. Robust Comment Fetching
    comments = []
    next_page_token = None
    target_count = 100 # Your new target
    
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        
        while len(comments) < target_count:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100, # Max allowed per request
                textFormat="plainText",
                order="relevance",
                pageToken=next_page_token # Tell YouTube which page to fetch
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': snippet['authorDisplayName'],
                    'text': snippet['textDisplay'],
                    'likes': snippet['likeCount']
                })
            
            # Update the token for the next loop
            next_page_token = response.get('nextPageToken')
            
            # If there are no more comments left on YouTube, break early
            if not next_page_token:
                break
                
    except Exception as e:
        print(f"Comment Fetching Error: {e}")
    
    return transcript, comments[:target_count]