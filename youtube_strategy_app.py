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
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Content Analysis", "🚀 Strategy", "📸 Thumbnails"])
                
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
                    if df is not None and not df.empty:
                        st.success(f"Analyzed {len(df)} videos (cached or freshly fetched).")
                        # Show analysis
                        tab1, tab2, tab3 = st.tabs(["📊 Performance", "🎯 Content Analysis", "🚀 Strategy"])
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
                    else:
                        st.error("Could not fetch channel data. Please check the URL, API key, and try again.")
    
    else:  # Upload Custom Data
        if 'uploaded_file' in locals() and uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded data with {len(df)} videos!")
            
            # Show analysis for uploaded data
            tab1, tab2, tab3 = st.tabs(["📊 Performance", "🎯 Content Analysis", "🚀 Strategy"])
            
            with tab1:
                create_performance_dashboard(df, "Uploaded Channel")
            
            with tab2:
                create_content_analysis(df)
            
            with tab3:
                create_strategy_recommendations(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Built with ❤️ for content creators**  
    *This tool analyzes successful YouTube channels to help you build your own winning content strategy.*
    
    💡 **Pro Tip**: Combine insights from multiple successful channels in your niche for best results!
    """)

if __name__ == "__main__":
    main()
