#!/usr/bin/env python3
"""
YouTube Channel Strategy Analyzer
A comprehensive Streamlit app that analyzes successful YouTube channels and provides 
content strategy insights for creators looking to build their own successful channels.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import re
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import ast
import base64
from io import BytesIO
import os
from typing import List, Dict, Any, Optional
import time
import hashlib
from urllib.parse import urlparse, parse_qs
import google.genai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
import random
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from Channel_analysis.youtube_fetcher import (
    fetch_channel_videos as fetch_channel_videos_api,
    normalize_channel_input,
    resolve_channel_id_from_web,
)
from supabase_service import (
    get_channel_analytics,
    upsert_channel_analytics,
    get_strategy,
    upsert_strategy,
    upsert_creator_persona,
    get_creator_persona,
    add_journal_entry,
    list_journal_entries,
    record_preference_event,
    list_preference_events,
    list_saved_channel_analytics,
)

# Load environment variables
load_dotenv()

def _get_first_available_secret(names, default=None):
    """Return the first non-empty value from Streamlit secrets or environment.

    Checks names in order across st.secrets then os.getenv.
    """
    for name in names:
        # Try Streamlit secrets if available
        try:
            if 'secrets' in dir(st) and name in st.secrets:
                value = st.secrets.get(name)
                if value:
                    return str(value)
        except Exception:
            pass
        # Fallback to environment variable
        value = os.getenv(name)
        if value:
            return value
    return default

# Configure API keys from Streamlit secrets or env vars
GEMINI_API_KEY = _get_first_available_secret([
    'GEMINI_API_KEY',
    'GOOGLE_API_KEY',
    'GOOGLE_GENAI_API_KEY',
])

# YouTube Data API key (supports multiple common names)
YOUTUBE_DATA_API_KEY = _get_first_available_secret([
    'YOUTUBE_DATA_API_KEY',
    'YOUTUBE_API_KEY',
])
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None

# Optional import for MVP kit persistence (fallback to no-op if missing)
try:
    import supabase_service as _sb
    save_mvp_video_kit = getattr(_sb, "save_mvp_video_kit", None)
except Exception:
    save_mvp_video_kit = None

# Page configuration
st.set_page_config(
    page_title="YouTube Strategy Analyzer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF0000, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class YouTubeAnalyzer:
    """Advanced YouTube channel analysis with AI-powered insights"""
    
    def __init__(self):
        self.client = client
        self.model = 'gemini-2.5-flash' if client else None
    
    def extract_channel_id(self, url: str) -> Optional[str]:
        """Extract channel ID from various YouTube URL formats"""
        patterns = [
            r'youtube\.com/channel/([a-zA-Z0-9_-]+)',
            r'youtube\.com/c/([a-zA-Z0-9_-]+)',
            r'youtube\.com/@([a-zA-Z0-9_-]+)',
            r'youtube\.com/user/([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_channel_videos_via_api(self, channel_input: str, api_key: str, max_videos: int = 50, include_comments: bool = True, comment_limit: int = 20) -> pd.DataFrame:
        """Fetch channel videos using YouTube Data API via Channel_analysis.youtube_fetcher."""
        try:
            df = fetch_channel_videos_api(
                channel_input,
                method='api',
                api_key=api_key,
                max_videos=max_videos,
                include_comments=include_comments,
                comment_limit=comment_limit,
                include_transcript=False,
                transcript_languages=None,
                transcript_max_chars=None,
            )
            return df
        except Exception as e:
            st.error(f"Error fetching via YouTube Data API: {e}")
            return pd.DataFrame()
    
    # yt-dlp based detail fetch removed
    
    def analyze_thumbnails(self, videos: List[Dict]) -> Dict[str, Any]:
        """Analyze thumbnail patterns and effectiveness"""
        if not self.model:
            return {"error": "Gemini API not available"}
            
        thumbnail_analysis = {
            'patterns': [],
            'color_schemes': [],
            'text_elements': [],
            'face_presence': 0,
            'recommendations': []
        }
        
        # Sample a few thumbnails for AI analysis
        sample_videos = videos[:10] if len(videos) > 10 else videos
        
        for video in sample_videos:
            if video.get('thumbnail'):
                try:
                    # Download and analyze thumbnail
                    response = requests.get(video['thumbnail'])
                    if response.status_code == 200:
                        # Convert to base64 for AI analysis
                        image_data = base64.b64encode(response.content).decode()
                        
                        prompt = """
                        Analyze this YouTube thumbnail and identify:
                        1. Color scheme (dominant colors)
                        2. Text elements (presence of text, style, readability)
                        3. Facial expressions or people present
                        4. Visual composition and layout
                        5. Emotional appeal and click-worthiness factors
                        
                        Provide a brief analysis in JSON format.
                        """
                        
                        # Note: In a real implementation, you'd send the image to Gemini
                        # For now, we'll do basic analysis
                        thumbnail_analysis['patterns'].append({
                            'video_id': video['video_id'],
                            'title': video['title'],
                            'views': video['view_count']
                        })
                        
                except Exception as e:
                    continue
        
        return thumbnail_analysis
    
    def analyze_content_patterns(self, videos: List[Dict]) -> Dict[str, Any]:
        """Analyze content patterns for success factors"""
        if not videos:
            return {}
            
        try:
            df = pd.DataFrame(videos)
        except Exception as e:
            st.error(f"Error creating DataFrame: {str(e)}")
            return {}
        
        # Handle different column names for upload date
        if 'published_at' in df.columns:
            df['upload_date'] = df['published_at']
        elif 'upload_date' not in df.columns:
            df['upload_date'] = ''
        
        # Performance metrics with error handling
        try:
            df['views_per_day'] = df.apply(
                lambda row: self._calculate_views_per_day(row.get('view_count', 0), row.get('upload_date', '')), 
                axis=1
            )
        except Exception as e:
            st.warning(f"Could not calculate views per day: {str(e)}")
            df['views_per_day'] = 0
        
        # Title analysis with error handling
        try:
            df['title_length'] = df['title'].fillna('').str.len()
            df['title_caps'] = df['title'].fillna('').apply(lambda x: sum(1 for c in str(x) if c.isupper()))
            df['title_numbers'] = df['title'].fillna('').apply(lambda x: len(re.findall(r'\d+', str(x))))
            df['title_exclamation'] = df['title'].fillna('').apply(lambda x: str(x).count('!'))
            df['title_question'] = df['title'].fillna('').apply(lambda x: str(x).count('?'))
        except Exception as e:
            st.warning(f"Could not analyze titles: {str(e)}")
            df['title_length'] = 0
            df['title_caps'] = 0
            df['title_numbers'] = 0
            df['title_exclamation'] = 0
            df['title_question'] = 0
        
        # Duration analysis
        # Handle different column names for duration
        if 'duration_seconds' in df.columns:
            df['duration_minutes'] = df['duration_seconds'] / 60
        elif 'duration' in df.columns:
            df['duration_minutes'] = df['duration'] / 60
        else:
            df['duration_minutes'] = 0
            
        df['duration_category'] = pd.cut(
            df['duration_minutes'], 
            bins=[0, 1, 5, 10, 20, float('inf')], 
            labels=['<1min', '1-5min', '5-10min', '10-20min', '20min+']
        )
        
        # Performance categories
        df['performance_tier'] = pd.qcut(
            df['view_count'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        return {
            'dataframe': df,
            'top_performers': df.nlargest(10, 'view_count'),
            'title_patterns': self._analyze_title_patterns(df),
            'duration_insights': self._analyze_duration_patterns(df),
            'performance_factors': self._identify_success_factors(df)
        }

    def _calculate_views_per_day(self, views: int, upload_date: str) -> float:
        """Calculate views per day since upload"""
        try:
            if not upload_date:
                return 0
            # Handle different date formats
            try:
                # Try ISO format first (from CSV: 2025-08-24 13:00:06+00:00)
                if 'T' in upload_date or ' ' in upload_date:
                    # Parse ISO format and strip timezone info
                    upload_date_clean = upload_date.split('+')[0].split('T')[0].replace(' ', 'T').split('T')[0]
                    upload_dt = datetime.strptime(upload_date_clean, '%Y-%m-%d')
                else:
                    # Try original format (YYYYMMDD)
                    upload_dt = datetime.strptime(upload_date, '%Y%m%d')
            except ValueError:
                # If both fail, try parsing as ISO with time
                upload_dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00').split('+')[0])
            days_since = (datetime.now() - upload_dt).days
            return views / max(days_since, 1)
        except:
            return 0

    def _analyze_title_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze title patterns for high-performing videos"""
        high_performers = df[df['performance_tier'] == 'High']
        return {
            'avg_length': high_performers['title_length'].mean(),
            'common_words': self._get_common_words(high_performers['title'].tolist()),
            'caps_usage': high_performers['title_caps'].mean(),
            'numbers_usage': high_performers['title_numbers'].mean(),
            'punctuation': {
                'exclamation': high_performers['title_exclamation'].mean(),
                'question': high_performers['title_question'].mean()
            }
        }

    def _analyze_duration_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duration patterns for engagement"""
        duration_performance = df.groupby('duration_category', observed=False).agg({
            'view_count': 'mean',
            'like_count': 'mean',
            'video_id': 'count'
        }).round(0)
        return duration_performance.to_dict()

    def _identify_success_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify key factors that correlate with success"""
        high_performers = df[df['performance_tier'] == 'High']
        correlations = df[[
            'view_count', 'title_length', 'title_caps', 'title_numbers',
            'title_exclamation', 'title_question', 'duration_minutes'
        ]].corr()['view_count'].abs().sort_values(ascending=False)
        return {
            'correlations': correlations.to_dict(),
            'high_performer_traits': {
                'avg_title_length': high_performers['title_length'].mean(),
                'avg_duration': high_performers['duration_minutes'].mean(),
                'common_title_words': self._get_common_words(high_performers['title'].tolist())
            }
        }

    def _get_common_words(self, titles: List[str]) -> List[str]:
        """Extract most common words from titles"""
        all_words = []
        for title in titles:
            words = re.findall(r'\b[a-zA-Z]+\b', title.lower())
            all_words.extend([word for word in words if len(word) > 3])
        word_counts = pd.Series(all_words).value_counts()
        return word_counts.head(10).index.tolist()

    def generate_content_strategy(self, channel_key: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered content strategy recommendations"""
        # First, try cached strategy from Supabase
        try:
            cached = get_strategy(channel_key)
            if isinstance(cached, dict) and cached.get('ai_recommendations'):
                return cached
        except Exception:
            pass
        if not self.client:
            basic = self._generate_basic_strategy(analysis_results)
            try:
                upsert_strategy(channel_key, basic)
            except Exception:
                pass
            return basic
        try:
            # Prepare analysis data for AI
            top_performers = analysis_results.get('top_performers', pd.DataFrame())
            title_patterns = analysis_results.get('title_patterns', {})
            duration_insights = analysis_results.get('duration_insights', {})
            prompt = f"""
            Based on the following YouTube channel analysis data, generate a comprehensive content format blueprint and strategy guide:

            Top Performing Videos:
            {top_performers[['title', 'view_count', 'duration_minutes']].head().to_string() if not top_performers.empty else 'No data available'}

            Title Patterns:
            - Average length: {title_patterns.get('avg_length', 'N/A')}
            - Common words: {title_patterns.get('common_words', [])}
            - Capitalization usage: {title_patterns.get('caps_usage', 'N/A')}

            Duration Insights:
            {duration_insights}

            Please provide a detailed blueprint that includes:

            1. **Content Planning Strategy**
            - Topic selection criteria based on top performers
            - Research methodology for similar content
            - Content themes that resonate with this audience

            2. **Video Structure Blueprint**
            - Optimal video length recommendations based on duration data
            - Segment organization patterns
            - Opening and closing techniques observed in top videos

            3. **Title and Thumbnail Optimization**
            - Title formulas derived from successful patterns
            - Capitalization and keyword strategies
            - Thumbnail design principles for this niche

            4. **Content Creation Guidelines**
            - Writing/scripting style recommendations
            - Delivery techniques and presentation methods
            - Production quality standards

            5. **Technical Specifications**
            - Audio/visual requirements
            - Editing and post-production guidelines
            - Publishing optimization

            6. **Growth and Engagement Tactics**
            - Posting frequency suggestions based on channel performance
            - Audience engagement strategies specific to this content type
            - Cross-video connection techniques

            7. **Implementation Checklist**
            - Pre-production planning steps
            - Production workflow
            - Post-production and publishing process

            Format this as a comprehensive, actionable blueprint that could be used by content creators to replicate this channel's successful format and approach. Include specific examples from the data where relevant, and make recommendations scalable for channels at different growth stages.
            """
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            generated = {
                'ai_recommendations': response.text,
                'strategy_type': 'ai_generated'
            }
            try:
                upsert_strategy(channel_key, generated)
            except Exception:
                pass
            return generated
        except Exception as e:
            basic = self._generate_basic_strategy(analysis_results)
            try:
                upsert_strategy(channel_key, basic)
            except Exception:
                pass
            return basic

    def _generate_basic_strategy(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic strategy recommendations without AI"""
        title_patterns = analysis_results.get('title_patterns', {})
        duration_insights = analysis_results.get('duration_insights', {})
        recommendations = [
            f"Optimal title length: {title_patterns.get('avg_length', 50):.0f} characters",
            f"Popular keywords: {', '.join(title_patterns.get('common_words', ['trending', 'tips', 'guide'])[:5])}",
            "Post consistently 2-3 times per week",
            "Focus on high-engagement video lengths (8-12 minutes)",
            "Use compelling thumbnails with bright colors and clear text",
            "Engage with comments within first hour of posting"
        ]
        return {
            'ai_recommendations': '\n'.join(recommendations),
            'strategy_type': 'basic'
        }

class CreatorPersonaModel(BaseModel):
    communication_style: str
    interests: List[str]
    creative_constraints: List[str]
    audience_relationship: str
    growth_priorities: str

class JournalEntryModel(BaseModel):
    persona_key: str
    channel_context: str
    content_title: str
    content_type: str
    decision: str
    time_spent_minutes: int
    emotions: List[str]
    plan_to_publish: bool
    published_url: str
    notes: str

class MinimalVideoKitModel(BaseModel):
    title: str
    hook: str
    outline: List[str]
    script: List[str]
    thumbnail_prompt: str
    description: str
    tags: List[str]
    duration_seconds: int

def _build_user_context_for_ai(persona_key: str) -> str:
    try:
        persona = get_creator_persona(persona_key) or {}
        entries = list_journal_entries(limit=10)
        entries = [e for e in entries if e.get('persona_key') == persona_key][:5] or entries[:5]
        lines = [
            f"Persona: comm_style={persona.get('communication_style','')}",
            f"Interests={', '.join(persona.get('interests', []))}",
            f"Constraints={', '.join(persona.get('creative_constraints', []))}",
            f"AudienceRel={persona.get('audience_relationship','')}",
            f"GrowthPriorities={persona.get('growth_priorities','')}"
        ]
        for e in entries:
            lines.append(
                f"Journal: title={e.get('content_title','')}, type={e.get('content_type','')}, decision={e.get('decision','')}, emotions={','.join(e.get('emotions',[]))}"
            )
        return "\n".join(lines)
    except Exception:
        return ""

def ai_fill_persona(persona_key: str, channel_context: Optional[str] = None) -> Optional[CreatorPersonaModel]:
    if not client:
        return None
    try:
        context = _build_user_context_for_ai(persona_key)
        prompt = f"""
        You are helping a YouTube creator define their Creative DNA. Based on the context below, produce a concise persona.
        Return only the JSON for these required fields. Keep lists short (3-6 items).
        Context:\n{context}\nChannelContext:{channel_context or ''}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CreatorPersonaModel,
            },
        )
        return getattr(response, 'parsed', None)
    except Exception:
        return None

def ai_suggest_journal_entry(persona_key: str, seed_title: str, channel_context: Optional[str] = None) -> Optional[JournalEntryModel]:
    if not client:
        return None
    try:
        context = _build_user_context_for_ai(persona_key)
        prompt = f"""
        Suggest a structured Content Journal entry for the creator. Use the seed title if provided.
        Keep emotions 1-3 items. Use realistic time_spent_minutes (5-120). Notes should be crisp (<= 280 chars).
        Return only JSON for required fields.
        Context:\n{context}\nSeedTitle:{seed_title}\nChannelContext:{channel_context or ''}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": JournalEntryModel,
            },
        )
        return getattr(response, 'parsed', None)
    except Exception:
        return None

def ai_fill_persona_from_text(context_text: str) -> Optional[CreatorPersonaModel]:
    if not client:
        return None
    try:
        prompt = f"""
        Derive a concise creator persona from the following transcript/context. Return only JSON with required fields.
        Context:\n{context_text[:12000]}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CreatorPersonaModel,
            },
        )
        return getattr(response, 'parsed', None)
    except Exception:
        return None

def _fetch_transcript_for_video(video_id: str) -> Optional[str]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([chunk.get('text', '') for chunk in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

def _random_top5_transcript(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df is None or df.empty:
            return None
        df_sorted = df.sort_values('view_count', ascending=False).head(5)
        if df_sorted.empty:
            return None
        row = df_sorted.sample(1).iloc[0]
        vid = str(row.get('video_id') or '')
        txt = _fetch_transcript_for_video(vid) if vid else None
        if not txt:
            # Fallback to title if transcript unavailable
            txt = str(row.get('title', ''))
        return {
            'video_id': vid,
            'title': row.get('title', ''),
            'transcript': txt or ''
        }
    except Exception:
        return None

def ai_generate_mvp_video(persona: Dict[str, Any], seed_texts: List[str]) -> Optional[MinimalVideoKitModel]:
    if not client:
        return None
    try:
        seed_blob = "\n".join([s for s in seed_texts if s])[:16000]
        prompt = f"""
        Generate a minimal viable short video kit for a YouTube creator.
        Base it on the persona and seed context below. Keep it tight and executable.
        Persona:\n{json.dumps(persona)[:4000]}
        Seed:\n{seed_blob}
        Constraints:
        - Duration 45-120 seconds.
        - Outline 4-6 bullets, Script 6-12 short lines.
        - Tags 5-10 items.
        - Hook punchy, single sentence. CTA specific.
        Return only JSON for required fields.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": MinimalVideoKitModel,
            },
        )
        return getattr(response, 'parsed', None)
    except Exception:
        return None

def _derive_persona_key(channel_key: Optional[str] = None) -> str:
    """Return a stable persona key based on environment or channel context."""
    base = os.getenv("CREATOR_ID") or os.getenv("USER") or "default_creator"
    if channel_key:
        return f"{base}:{hashlib.sha256(channel_key.encode()).hexdigest()[:12]}"
    return base

def render_creator_persona_section(persona_key: str, df: Optional[pd.DataFrame] = None, auto_generate: bool = True):
    st.subheader("🧬 Creator Persona (Auto-generated from top video transcript)")
    existing = get_creator_persona(persona_key) or {}
    if auto_generate and not existing:
        sample = _random_top5_transcript(df) if df is not None else None
        transcript_text = sample.get('transcript', '') if sample else ''
        parsed = ai_fill_persona_from_text(transcript_text) if transcript_text else None
        if parsed:
            persona = parsed.model_dump()
            try:
                upsert_creator_persona(persona_key, persona)
                existing = persona
                st.success("Persona generated and saved from transcript")
            except Exception as e:
                st.warning(f"Could not save persona: {e}")
        else:
            st.info("Transcript not available; using heuristic persona from titles.")
            titles_blob = " ".join((df.sort_values('view_count', ascending=False).head(5)['title'].astype(str).tolist())) if df is not None else ""
            parsed = ai_fill_persona_from_text(titles_blob)
            if parsed:
                persona = parsed.model_dump()
                try:
                    upsert_creator_persona(persona_key, persona)
                    existing = persona
                    st.success("Persona generated from titles and saved")
                except Exception:
                    pass
    col = st.container()
    with col:
        st.markdown("Generated Persona:")
        st.json(existing or {})
        if st.button("Regenerate from random top-5 transcript", key=f"regen_persona_{persona_key}"):
            sample = _random_top5_transcript(df) if df is not None else None
            transcript_text = sample.get('transcript', '') if sample else ''
            parsed = ai_fill_persona_from_text(transcript_text) if transcript_text else None
            if parsed:
                persona = parsed.model_dump()
                try:
                    upsert_creator_persona(persona_key, persona)
                    st.success("Persona regenerated and saved")
                    st.experimental_rerun()
                except Exception as e:
                    st.warning(f"Could not save persona: {e}")

def render_content_journal_section(persona_key: str, channel_context: Optional[str] = None):
    st.subheader("📓 Content Journal")
    st.caption("Track what you think/feel about each piece of content. Your system learns from this.")
    # AI suggest helper
    ai_cols = st.columns(2)
    with ai_cols[0]:
        seed_title = st.text_input("Seed Title/Concept for AI Suggest", key=f"journal_seed_{persona_key}")
    with ai_cols[1]:
        if st.button("🤖 Suggest Journal Entry", key=f"ai_suggest_journal_{persona_key}"):
            parsed = ai_suggest_journal_entry(persona_key, seed_title, channel_context)
            if parsed:
                st.session_state[f"journal_title_{persona_key}"] = parsed.content_title
                st.session_state[f"journal_type_{persona_key}"] = parsed.content_type
                st.session_state[f"journal_decision_{persona_key}"] = parsed.decision
                st.session_state[f"journal_time_{persona_key}"] = parsed.time_spent_minutes
                st.session_state[f"journal_emotions_{persona_key}"] = parsed.emotions
                st.session_state[f"journal_publish_{persona_key}"] = parsed.plan_to_publish
                st.session_state[f"journal_url_{persona_key}"] = parsed.published_url
                st.session_state[f"journal_notes_{persona_key}"] = parsed.notes
                st.success("AI suggested a journal entry. Review and submit.")
    with st.form(key=f"journal_form_{persona_key}"):
        col1, col2 = st.columns(2)
        with col1:
            content_title = st.text_input("Content Title/Idea", value=st.session_state.get(f"journal_title_{persona_key}", ""))
            content_type = st.selectbox("Content Type", ["Short", "Long-form", "Community Post", "Script", "Other"], index=(["Short", "Long-form", "Community Post", "Script", "Other"].index(st.session_state.get(f"journal_type_{persona_key}", "Short")) if st.session_state.get(f"journal_type_{persona_key}") in ["Short", "Long-form", "Community Post", "Script", "Other"] else 0))
            decision_options = ["generated", "modified", "rejected", "published"]
            decision_default = st.session_state.get(f"journal_decision_{persona_key}", decision_options[0])
            decision = st.selectbox("Decision", decision_options, index=(decision_options.index(decision_default) if decision_default in decision_options else 0)) 
            time_spent = st.number_input("Time Spent (minutes)", min_value=0, step=5, value=int(st.session_state.get(f"journal_time_{persona_key}", 0)))
        with col2:
            emotions = st.multiselect("Emotions", ["excited", "curious", "anxious", "confused", "fulfilled", "bored", "frustrated"], default=st.session_state.get(f"journal_emotions_{persona_key}", [])) 
            plan_to_publish = st.checkbox("Plan to publish / published", value=bool(st.session_state.get(f"journal_publish_{persona_key}", False)))
            published_url = st.text_input("Published URL (optional)", value=st.session_state.get(f"journal_url_{persona_key}", ""))
        notes = st.text_area("Notes / Why? / What changed?", value=st.session_state.get(f"journal_notes_{persona_key}", ""))
        submitted = st.form_submit_button("Add Journal Entry")
        if submitted:
            entry = {
                'persona_key': persona_key,
                'channel_context': channel_context or '',
                'content_title': content_title,
                'content_type': content_type,
                'decision': decision,
                'time_spent_minutes': int(time_spent),
                'emotions': emotions,
                'plan_to_publish': bool(plan_to_publish),
                'published_url': published_url,
                'notes': notes,
            }
            try:
                add_journal_entry(entry)
                st.success("Journal entry saved")
            except Exception as e:
                st.warning(f"Could not save entry: {e}")
    # List recent entries (filtered by persona_key if present)
    try:
        recent = list_journal_entries(limit=50)
        filtered = [e for e in recent if e.get('persona_key') == persona_key] or recent[:10]
        with st.expander("Recent Entries"):
            for e in filtered[:10]:
                st.markdown(f"**{e.get('content_title','(untitled)')}** · {e.get('content_type','')} · {e.get('decision','')}")
                st.caption(f"Emotions: {', '.join(e.get('emotions', []))} · Time: {e.get('time_spent_minutes',0)} min")
                if e.get('published_url'):
                    st.markdown(f"Published: {e.get('published_url')}")
                if e.get('notes'):
                    st.write(e.get('notes'))
                st.markdown("---")
    except Exception:
        pass

def render_mvp_video_section(df: pd.DataFrame, persona_key: str, channel_key: Optional[str] = None):
    st.subheader("🎬 Minimal Viable Video (Auto-generated)")
    persona = get_creator_persona(persona_key) or {}
    sample = _random_top5_transcript(df)
    transcript_text = sample.get('transcript', '') if sample else ''
    seed_texts = [transcript_text]
    kit = ai_generate_mvp_video(persona, seed_texts)
    if not kit:
        st.info("AI unavailable or generation failed.")
        return
    # Persist generated kit to Supabase
    try:
        kit_dict = kit.model_dump() if hasattr(kit, 'model_dump') else {
            'title': getattr(kit, 'title', None),
            'hook': getattr(kit, 'hook', None),
            'outline': getattr(kit, 'outline', None),
            'script': getattr(kit, 'script', None),
            'thumbnail_prompt': getattr(kit, 'thumbnail_prompt', None),
            'description': getattr(kit, 'description', None),
            'tags': getattr(kit, 'tags', None),
            'duration_seconds': getattr(kit, 'duration_seconds', None),
        }
        if callable(save_mvp_video_kit):
            save_mvp_video_kit(channel_key or 'uploaded', persona_key, kit_dict)
    except Exception:
        pass
    st.markdown(f"**Title**: {kit.title}")
    st.markdown(f"**Hook**: {kit.hook}")
    st.markdown("**Outline**:")
    for b in kit.outline:
        st.markdown(f"- {b}")
    st.markdown("**Script**:")
    for line in kit.script:
        st.markdown(f"- {line}")
    st.markdown(f"**Thumbnail Prompt**: {kit.thumbnail_prompt}")
    st.markdown(f"**Description**: {kit.description}")
    st.markdown(f"**Tags**: {', '.join(kit.tags)}")
    st.caption(f"Estimated Duration: {kit.duration_seconds} sec")
    # Download as Markdown
    md = [
        f"# {kit.title}",
        f"Hook: {kit.hook}",
        "\n## Outline",
    ] + [f"- {b}" for b in kit.outline] + [
        "\n## Script",
    ] + [f"- {s}" for s in kit.script] + [
        f"\nThumbnail Prompt: {kit.thumbnail_prompt}",
        f"\nDescription: {kit.description}",
        f"\nTags: {', '.join(kit.tags)}",
        f"\nDuration: {kit.duration_seconds} sec",
    ]
    md_blob = "\n".join(md)
    st.download_button("Download Kit (Markdown)", data=md_blob, file_name="mvp_video_kit.md")
    if st.button("Regenerate Kit", key=f"regen_kit_{channel_key or 'uploaded'}"):
        st.experimental_rerun()

def _format_published_date(value: Any) -> str:
    """Return a safe YYYY-MM-DD string for various input types."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    # If it's already a pandas/py datetime
    try:
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
    except Exception:
        pass
    if isinstance(value, datetime):
        return value.date().isoformat()
    # If it's a string, try to parse or slice safely
    if isinstance(value, str):
        try:
            dt = pd.to_datetime(value, errors='coerce')
            if pd.notna(dt):
                return dt.date().isoformat()
        except Exception:
            pass
        # Fallback: best-effort first 10 chars if looks like ISO
        return value[:10]
    # Last resort
    return str(value)[:10]
    
    def _calculate_views_per_day(self, views: int, upload_date: str) -> float:
        """Calculate views per day since upload"""
        try:
            if not upload_date:
                return 0
            
            # Handle different date formats
            try:
                # Try ISO format first (from CSV: 2025-08-24 13:00:06+00:00)
                if 'T' in upload_date or ' ' in upload_date:
                    # Parse ISO format and strip timezone info
                    upload_date_clean = upload_date.split('+')[0].split('T')[0].replace(' ', 'T').split('T')[0]
                    upload_dt = datetime.strptime(upload_date_clean, '%Y-%m-%d')
                else:
                    # Try original format (YYYYMMDD)
                    upload_dt = datetime.strptime(upload_date, '%Y%m%d')
            except ValueError:
                # If both fail, try parsing as ISO with time
                upload_dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00').split('+')[0])
            
            days_since = (datetime.now() - upload_dt).days
            return views / max(days_since, 1)
        except:
            return 0
    
    def _analyze_title_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze title patterns for high-performing videos"""
        high_performers = df[df['performance_tier'] == 'High']
        
        return {
            'avg_length': high_performers['title_length'].mean(),
            'common_words': self._get_common_words(high_performers['title'].tolist()),
            'caps_usage': high_performers['title_caps'].mean(),
            'numbers_usage': high_performers['title_numbers'].mean(),
            'punctuation': {
                'exclamation': high_performers['title_exclamation'].mean(),
                'question': high_performers['title_question'].mean()
            }
        }
    
    def _analyze_duration_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duration patterns for engagement"""
        duration_performance = df.groupby('duration_category', observed=False).agg({
            'view_count': 'mean',
            'like_count': 'mean',
            'video_id': 'count'
        }).round(0)
        
        return duration_performance.to_dict()
    
    def _identify_success_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify key factors that correlate with success"""
        high_performers = df[df['performance_tier'] == 'High']
        
        correlations = df[[
            'view_count', 'title_length', 'title_caps', 'title_numbers',
            'title_exclamation', 'title_question', 'duration_minutes'
        ]].corr()['view_count'].abs().sort_values(ascending=False)
        
        return {
            'correlations': correlations.to_dict(),
            'high_performer_traits': {
                'avg_title_length': high_performers['title_length'].mean(),
                'avg_duration': high_performers['duration_minutes'].mean(),
                'common_title_words': self._get_common_words(high_performers['title'].tolist())
            }
        }
    
    def _get_common_words(self, titles: List[str]) -> List[str]:
        """Extract most common words from titles"""
        all_words = []
        for title in titles:
            words = re.findall(r'\b[a-zA-Z]+\b', title.lower())
            all_words.extend([word for word in words if len(word) > 3])
        
        word_counts = pd.Series(all_words).value_counts()
        return word_counts.head(10).index.tolist()
    
    def generate_content_strategy(self, channel_key: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered content strategy recommendations"""
        # First, try cached strategy from Supabase
        try:
            cached = get_strategy(channel_key)
            if isinstance(cached, dict) and cached.get('ai_recommendations'):
                return cached
        except Exception:
            pass
        if not self.client:
            basic = self._generate_basic_strategy(analysis_results)
            try:
                upsert_strategy(channel_key, basic)
            except Exception:
                pass
            return basic
            
        try:
            # Prepare analysis data for AI
            top_performers = analysis_results.get('top_performers', pd.DataFrame())
            title_patterns = analysis_results.get('title_patterns', {})
            duration_insights = analysis_results.get('duration_insights', {})
            
            prompt = f"""
            Based on the following YouTube channel analysis data, generate a comprehensive content format blueprint and strategy guide:

            Top Performing Videos:
            {top_performers[['title', 'view_count', 'duration_minutes']].head().to_string() if not top_performers.empty else 'No data available'}

            Title Patterns:
            - Average length: {title_patterns.get('avg_length', 'N/A')}
            - Common words: {title_patterns.get('common_words', [])}
            - Capitalization usage: {title_patterns.get('caps_usage', 'N/A')}

            Duration Insights:
            {duration_insights}

            Please provide a detailed blueprint that includes:

            1. **Content Planning Strategy**
            - Topic selection criteria based on top performers
            - Research methodology for similar content
            - Content themes that resonate with this audience

            2. **Video Structure Blueprint**
            - Optimal video length recommendations based on duration data
            - Segment organization patterns
            - Opening and closing techniques observed in top videos

            3. **Title and Thumbnail Optimization**
            - Title formulas derived from successful patterns
            - Capitalization and keyword strategies
            - Thumbnail design principles for this niche

            4. **Content Creation Guidelines**
            - Writing/scripting style recommendations
            - Delivery techniques and presentation methods
            - Production quality standards

            5. **Technical Specifications**
            - Audio/visual requirements
            - Editing and post-production guidelines
            - Publishing optimization

            6. **Growth and Engagement Tactics**
            - Posting frequency suggestions based on channel performance
            - Audience engagement strategies specific to this content type
            - Cross-video connection techniques

            7. **Implementation Checklist**
            - Pre-production planning steps
            - Production workflow
            - Post-production and publishing process

            Format this as a comprehensive, actionable blueprint that could be used by content creators to replicate this channel's successful format and approach. Include specific examples from the data where relevant, and make recommendations scalable for channels at different growth stages.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            generated = {
                'ai_recommendations': response.text,
                'strategy_type': 'ai_generated'
            }
            try:
                upsert_strategy(channel_key, generated)
            except Exception:
                pass
            return generated
            
        except Exception as e:
            basic = self._generate_basic_strategy(analysis_results)
            try:
                upsert_strategy(channel_key, basic)
            except Exception:
                pass
            return basic
    
    def _generate_basic_strategy(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic strategy recommendations without AI"""
        title_patterns = analysis_results.get('title_patterns', {})
        duration_insights = analysis_results.get('duration_insights', {})
        
        recommendations = [
            f"Optimal title length: {title_patterns.get('avg_length', 50):.0f} characters",
            f"Popular keywords: {', '.join(title_patterns.get('common_words', ['trending', 'tips', 'guide'])[:5])}",
            "Post consistently 2-3 times per week",
            "Focus on high-engagement video lengths (8-12 minutes)",
            "Use compelling thumbnails with bright colors and clear text",
            "Engage with comments within first hour of posting"
        ]
        
        return {
            'ai_recommendations': '\n'.join(recommendations),
            'strategy_type': 'basic'
        }

def load_sample_data():
    """Load sample channel data from existing CSV files"""
    data_dir = Path("Channel_analysis/outputs/channels")
    channels_data = {}
    
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                channel_name = csv_file.stem.replace("at_", "@").replace("www.youtube.com_", "")
                channels_data[channel_name] = df
            except Exception as e:
                st.warning(f"Could not load {csv_file}: {e}")
    
    # Merge in channels saved in Supabase (most recent first)
    try:
        saved = list_saved_channel_analytics(limit=50)
        for key, df in saved.items():
            # Prefer human-readable key if present in DF
            readable = None
            try:
                if 'channel' in df.columns and isinstance(df.iloc[0]['channel'], str):
                    readable = str(df.iloc[0]['channel'])
            except Exception:
                readable = None
            name = readable or key
            if name not in channels_data and isinstance(df, pd.DataFrame) and not df.empty:
                channels_data[name] = df
    except Exception:
        pass

    return channels_data

def create_performance_dashboard(df: pd.DataFrame, channel_name: str):
    """Create comprehensive performance dashboard"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Videos</h3>
            <h2>{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_views = df['view_count'].mean()
        st.markdown("""
        <div class="success-card">
            <h3>Avg Views</h3>
            <h2>{:,.0f}</h2>
        </div>
        """.format(avg_views), unsafe_allow_html=True)
    
    with col3:
        total_views = df['view_count'].sum()
        st.markdown("""
        <div class="metric-card">
            <h3>Total Views</h3>
            <h2>{:,.0f}</h2>
        </div>
        """.format(total_views), unsafe_allow_html=True)
    
    with col4:
        avg_duration = df['duration_seconds'].mean() / 60
        st.markdown("""
        <div class="warning-card">
            <h3>Avg Duration</h3>
            <h2>{:.1f} min</h2>
        </div>
        """.format(avg_duration), unsafe_allow_html=True)
    
    # Performance over time
    st.subheader("📈 Performance Trends")
    
    # Convert published_at to datetime
    df['published_date'] = pd.to_datetime(df['published_at'], errors='coerce')
    df_sorted = df.sort_values('published_date')
    
    # Views over time
    fig_views = px.line(
        df_sorted, 
        x='published_date', 
        y='view_count',
        title='Views Over Time',
        color_discrete_sequence=['#FF4500']
    )
    fig_views.update_layout(height=400)
    st.plotly_chart(fig_views, use_container_width=True)
    
    # Duration vs Performance
    col1, col2 = st.columns(2)
    
    with col1:
        fig_duration = px.scatter(
            df, 
            x='duration_seconds', 
            y='view_count',
            hover_data=['title'],
            title='Duration vs Views',
            color='like_count',
            color_continuous_scale='Viridis'
        )
        fig_duration.update_layout(height=400)
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Title length analysis
        df['title_length'] = df['title'].str.len()
        fig_title = px.scatter(
            df,
            x='title_length',
            y='view_count',
            hover_data=['title'],
            title='Title Length vs Views',
            color='duration_seconds',
            color_continuous_scale='Plasma'
        )
        fig_title.update_layout(height=400)
        st.plotly_chart(fig_title, use_container_width=True)

def create_content_analysis(df: pd.DataFrame):
    """Analyze content patterns and themes"""
    st.subheader("🎯 Content Analysis")
    
    # Top performing videos
    st.subheader("🏆 Top Performers")
    top_videos = df.nlargest(10, 'view_count')[['title', 'view_count', 'like_count', 'published_at', 'url']]
    
    for idx, (_, video) in enumerate(top_videos.iterrows(), 1):
        with st.expander(f"#{idx} - {video['title'][:60]}..."):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Views", f"{video['view_count']:,}")
            with col2:
                st.metric("Likes", f"{video['like_count']:,}")
            with col3:
                st.metric("Published", _format_published_date(video['published_at']))
            st.markdown(f"[Watch Video]({video['url']})")
    
    # Word cloud data
    all_titles = ' '.join(df['title'].astype(str))
    words = re.findall(r'\b[a-zA-Z]+\b', all_titles.lower())
    word_freq = pd.Series([word for word in words if len(word) > 3]).value_counts().head(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Most Common Title Words")
        fig_words = px.bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            title='Top 20 Words in Titles',
            color=word_freq.values,
            color_continuous_scale='Blues'
        )
        fig_words.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_words, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Duration Distribution")
        df['duration_minutes'] = df['duration_seconds'] / 60
        duration_bins = pd.cut(df['duration_minutes'], bins=[0, 5, 10, 15, 30, float('inf')], labels=['0-5min', '5-10min', '10-15min', '15-30min', '30min+'])
        duration_counts = duration_bins.value_counts()
        
        fig_duration = px.pie(
            values=duration_counts.values,
            names=duration_counts.index,
            title='Video Duration Distribution'
        )
        st.plotly_chart(fig_duration, use_container_width=True)

def create_strategy_recommendations(df: pd.DataFrame, channel_key: Optional[str] = None):
    """Generate comprehensive strategy recommendations with Supabase caching."""
    st.subheader("🚀 Content Strategy Recommendations")
    analyzer = YouTubeAnalyzer()
    # Determine a stable channel_key when not provided
    if not channel_key:
        try:
            if 'channel' in df.columns and isinstance(df.iloc[0]['channel'], str):
                channel_key = str(df.iloc[0]['channel'])
            elif 'url' in df.columns and isinstance(df.iloc[0]['url'], str):
                channel_key = df.iloc[0]['url'].split('/watch?v=')[0]
            else:
                concat = '|'.join(map(str, df.get('video_id', pd.Series([])).tolist()[:20]))
                channel_key = f"uploaded:{hashlib.sha256(concat.encode()).hexdigest()[:16]}"
        except Exception:
            channel_key = f"uploaded:{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}"
    videos_data = df.to_dict('records')
    analysis_results = analyzer.analyze_content_patterns(videos_data)
    strategy = analyzer.generate_content_strategy(channel_key, analysis_results)
    # Optionally persist analytics snapshot as well
    try:
        upsert_channel_analytics(channel_key, df)
    except Exception:
        pass
    # Display recommendations in organized sections
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Title Optimization")
        title_patterns = analysis_results.get('title_patterns', {})
        st.info(f"""
        **Optimal Title Length**: {title_patterns.get('avg_length', 50):.0f} characters
        
        **High-Performing Keywords**: {', '.join(title_patterns.get('common_words', ['AI', 'Tech', 'Future'])[:5])}
        
        **Punctuation Usage**: Use exclamation marks and questions strategically
        """)
        st.markdown("### 📏 Video Length Strategy")
        try:
            avg_duration = (df['duration_seconds'].mean() or 0) / 60
            top_performers_avg = (df.nlargest(10, 'view_count')['duration_seconds'].mean() or 0) / 60
        except Exception:
            avg_duration = 0
            top_performers_avg = 0
        st.success(f"""
        **Channel Average**: {avg_duration:.1f} minutes
        
        **Top Performers Average**: {top_performers_avg:.1f} minutes
        
        **Recommendation**: Aim for {top_performers_avg:.0f}-{top_performers_avg*1.2:.0f} minute videos
        """)
    with col2:
        st.markdown("### 📅 Publishing Strategy")
        try:
            df['published_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
            if df['published_date'].dt.tz is not None:
                df['published_date_naive'] = df['published_date'].dt.tz_localize(None)
            else:
                df['published_date_naive'] = df['published_date']
            cutoff_date = datetime.now() - timedelta(days=90)
            df_recent = df[df['published_date_naive'] > cutoff_date]
            weekly_posts = len(df_recent) / 13
        except Exception as e:
            st.warning(f"Could not analyze posting frequency: {str(e)}")
            weekly_posts = 0
        st.warning(f"""
        **Current Frequency**: {weekly_posts:.1f} videos per week
        
        **Recommended**: 2-3 videos per week for optimal growth
        
        **Best Publishing Days**: Tuesday, Thursday, Saturday
        """)
        st.markdown("### 🎨 Visual Strategy")
        st.info("""
        **Thumbnail Tips**:
        - Use bright, contrasting colors
        - Include human faces when possible
        - Add clear, readable text
        - Maintain consistent branding
        
        **A/B Test**: Different thumbnail styles monthly
        """)
    if strategy.get('strategy_type') == 'ai_generated':
        st.markdown("### 🤖 AI-Powered Insights")
        with st.expander("View Detailed AI Recommendations"):
            st.markdown(strategy.get('ai_recommendations', ''))
    else:
        st.info(
            strategy.get('ai_recommendations', "AI insights are disabled. To enable, set GEMINI_API_KEY/GOOGLE_API_KEY/GOOGLE_GENAI_API_KEY.")
        )
    # Feedback capture for preference learning
    st.markdown("### 🧠 Feedback")
    rating = st.radio("Was this helpful?", ["Helpful", "Neutral", "Not helpful"], horizontal=True)
    reason = st.text_input("Optional: Why? What would improve it?")
    if st.button("Submit Feedback", key=f"feedback_{channel_key or 'uploaded'}"):
        try:
            strat_text = strategy.get('ai_recommendations', '')
            strategy_id = hashlib.sha256((str(channel_key) + '|' + strat_text).encode()).hexdigest()[:16]
            record_preference_event({
                'type': 'strategy_feedback',
                'channel_key': channel_key or 'uploaded',
                'strategy_id': strategy_id,
                'rating': rating,
                'reason': reason,
            })
            st.success("Thanks! Your feedback was recorded and will inform future suggestions.")
        except Exception as e:
            st.warning(f"Could not record feedback: {e}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📺 YouTube Strategy Analyzer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Analyze successful YouTube channels, decode their strategies, and build your own winning content plan.**
    
    Upload channel data or analyze successful channels to discover the secrets behind viral content.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Analysis Options")
        
        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            ["Sample Channels", "Analyze New Channel", "Upload Custom Data"]
        )
        
        if analysis_mode == "Sample Channels":
            st.info("Analyze pre-loaded successful AI/Tech channels")
            
        elif analysis_mode == "Analyze New Channel":
            st.info("Fetch and analyze any YouTube channel")
            channel_url = st.text_input(
                "YouTube Channel URL",
                placeholder="https://www.youtube.com/@channelname"
            )
            max_videos = st.slider("Max Videos to Analyze", 10, 100, 50)
            default_key = YOUTUBE_DATA_API_KEY or os.getenv("YOUTUBE_API_KEY", "")
            yt_api_key = st.text_input("YouTube Data API Key", value=default_key, type="password")
            
        else:
            st.info("Upload your own channel data CSV")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    # Main content area
    if analysis_mode == "Sample Channels":
        # Load sample data
        channels_data = load_sample_data()
        
        if channels_data:
            selected_channel = st.selectbox(
                "Select Channel to Analyze",
                list(channels_data.keys())
            )
            
            if selected_channel:
                df = channels_data[selected_channel]
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Performance", "🎯 Content Analysis", "🚀 Strategy", "📸 Thumbnails", "🧬 Persona", "📓 Journal", "🎬 MVP Video"])
                
                with tab1:
                    create_performance_dashboard(df, selected_channel)
                
                with tab2:
                    create_content_analysis(df)
                
                with tab3:
                    create_strategy_recommendations(df)
                
                with tab4:
                    st.subheader("🖼️ Thumbnail Analysis")
                    st.info("Thumbnail analysis coming soon! This will include AI-powered visual pattern recognition.")
                    
                    # Show sample thumbnails
                    if not df.empty:
                        st.subheader("Recent Video Thumbnails")
                        sample_videos = df.head(6)
                        
                        cols = st.columns(3)
                        for idx, (_, video) in enumerate(sample_videos.iterrows()):
                            with cols[idx % 3]:
                                st.markdown(f"**{video['title'][:40]}...**")
                                st.caption(f"Views: {video['view_count']:,}")
                                # In a real implementation, you'd display actual thumbnails here
                                st.markdown(f"[View Video]({video.get('url', '#')})")
                with tab5:
                    persona_key = _derive_persona_key(selected_channel)
                    render_creator_persona_section(persona_key, df=df, auto_generate=True)
                with tab6:
                    persona_key = _derive_persona_key(selected_channel)
                    render_content_journal_section(persona_key, channel_context=selected_channel)
                with tab7:
                    persona_key = _derive_persona_key(selected_channel)
                    render_mvp_video_section(df, persona_key, channel_key=selected_channel)
        else:
            st.warning("No sample data available. Please ensure channel data is in the Channel_analysis/outputs/channels/ directory.")
    
    elif analysis_mode == "Analyze New Channel":
        if st.button("🔍 Analyze Channel") and channel_url:
            if not yt_api_key:
                st.error("YouTube Data API Key is required.")
            else:
                with st.spinner("Fetching channel data via YouTube Data API... This may take a few minutes."):
                    analyzer = YouTubeAnalyzer()
                    # Compute a stable channel key for caching
                    try:
                        normalized_url = normalize_channel_input(channel_url)
                    except Exception:
                        normalized_url = channel_url.strip()
                    channel_id = None
                    try:
                        channel_id = resolve_channel_id_from_web(normalized_url)
                    except Exception:
                        channel_id = None
                    channel_key = channel_id or normalized_url
                    # Try Supabase cache first
                    df_cached = get_channel_analytics(channel_key)
                    if df_cached is not None and not df_cached.empty:
                        df = df_cached
                    else:
                        df = analyzer.get_channel_videos_via_api(channel_url, yt_api_key, max_videos=max_videos)
                        if df is not None and not df.empty:
                            try:
                                upsert_channel_analytics(channel_key, df)
                            except Exception:
                                pass
                    # Add to in-session sample list by reloading samples (includes DB)
                    try:
                        st.session_state["loaded_channels_cache"] = load_sample_data()
                    except Exception:
                        pass
                    if df is not None and not df.empty:
                        st.success(f"Analyzed {len(df)} videos (cached or freshly fetched).")
                        # Show analysis
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Performance", "🎯 Content Analysis", "🚀 Strategy", "🧬 Persona", "📓 Journal", "🎬 MVP Video"])
                        with tab1:
                            create_performance_dashboard(df, "Analyzed Channel")
                        with tab2:
                            create_content_analysis(df)
                        with tab3:
                            # Pass channel_key for strategy caching
                            analysis_results = YouTubeAnalyzer().analyze_content_patterns(df.to_dict('records'))
                            strategy = YouTubeAnalyzer().generate_content_strategy(channel_key, analysis_results)
                            # Render similar to create_strategy_recommendations but using cached strategy
                            st.markdown("### 🚀 Strategy Recommendations")
                            if strategy.get('strategy_type') == 'ai_generated':
                                st.markdown("### 🤖 AI-Powered Insights")
                                with st.expander("View Detailed AI Recommendations"):
                                    st.markdown(strategy.get('ai_recommendations', ''))
                            else:
                                st.info(strategy.get('ai_recommendations', ''))
                        with tab4:
                            persona_key = _derive_persona_key(channel_key)
                            render_creator_persona_section(persona_key, df=df, auto_generate=True)
                        with tab5:
                            persona_key = _derive_persona_key(channel_key)
                            render_content_journal_section(persona_key, channel_context=channel_key)
                        with tab6:
                            persona_key = _derive_persona_key(channel_key)
                            render_mvp_video_section(df, persona_key, channel_key=channel_key)
                    else:
                        st.error("Could not fetch channel data. Please check the URL, API key, and try again.")
    
    else:  # Upload Custom Data
        if 'uploaded_file' in locals() and uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded data with {len(df)} videos!")
            
            # Show analysis for uploaded data
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Performance", "🎯 Content Analysis", "🚀 Strategy", "🧬 Persona", "📓 Journal", "🎬 MVP Video"])
            
            with tab1:
                create_performance_dashboard(df, "Uploaded Channel")
            
            with tab2:
                create_content_analysis(df)
            
            with tab3:
                create_strategy_recommendations(df)
            with tab4:
                persona_key = _derive_persona_key("uploaded")
                render_creator_persona_section(persona_key, df=df, auto_generate=True)
            with tab5:
                persona_key = _derive_persona_key("uploaded")
                render_content_journal_section(persona_key, channel_context="uploaded")
            with tab6:
                persona_key = _derive_persona_key("uploaded")
                render_mvp_video_section(df, persona_key, channel_key="uploaded")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Built with ❤️ for content creators**  
    *This tool analyzes successful YouTube channels to help you build your own winning content strategy.*
    
    💡 **Pro Tip**: Combine insights from multiple successful channels in your niche for best results!
    """)

if __name__ == "__main__":
    main()
