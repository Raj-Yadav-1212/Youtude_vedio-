import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load Environment Variables
load_dotenv()
# LangChain looks for HUGGINGFACEHUB_API_TOKEN specifically
api = os.getenv("HF_TOKEN")

# 2. Initialize the Hugging Face Endpoint
# We use Mistral-7B-Instruct-v0.3 which is great for summaries
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.4,
)

# 3. Wrap in the Chat interface
chat_model = ChatHuggingFace(llm=llm)

def generate_summary(transcript_text):
    if not transcript_text or "not available" in transcript_text.lower():
        return "No transcript available for summary."

    try:
        # Split text into 3000-character chunks to avoid token limits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
        chunks = text_splitter.split_text(transcript_text)

        partial_summaries = []
        
        for chunk in chunks:
            messages = [
                SystemMessage(content="You are an expert summarizer. Summarize this transcript segment concisely."),
                HumanMessage(content=chunk)
            ]
            # This calls Hugging Face directly, no Together AI needed
            response = chat_model.invoke(messages)
            partial_summaries.append(response.content)

        full_summary = "\n\n".join(partial_summaries)
        
        # Final pass if the combined text is too long
        if len(full_summary) > 2000:
            final_messages = [
                SystemMessage(content="Consolidate these notes into a final 250-word cohesive summary."),
                HumanMessage(content=full_summary[:4000])
            ]
            final_response = chat_model.invoke(final_messages)
            return final_response.content
        
        return full_summary

    except Exception as e:
        return f"❌ Hugging Face Summary Error: {str(e)}"