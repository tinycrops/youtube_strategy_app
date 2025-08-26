#!/usr/bin/env python3
"""
Video Analyzer Integration for YouTube Strategy App
Connects the existing video analysis pipeline with the strategy analyzer
"""

import sys
import os
import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import requests
from datetime import datetime
import tempfile

# Import from the main app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import VideoAnalyzer, VideoDownloader, init_db, save_video, save_analysis, get_video_by_url

class StrategyVideoAnalyzer:
    """Enhanced video analyzer for strategy insights"""
    
    def __init__(self):
        self.base_analyzer = VideoAnalyzer()
        self.downloader = VideoDownloader()
        init_db()  # Ensure database is initialized
    
    def analyze_channel_videos(self, channel_videos: List[Dict], sample_size: int = 5) -> Dict[str, Any]:
        """Analyze a sample of channel videos for content insights"""
        
        results = {
            'analyzed_videos': [],
            'content_themes': [],
            'success_patterns': [],
            'ai_insights': [],
            'errors': []
        }
        
        # Select diverse sample (high, medium, low performers)
        if len(channel_videos) > sample_size:
            sorted_videos = sorted(channel_videos, key=lambda x: x.get('view_count', 0), reverse=True)
            sample_videos = []
            
            # Get top performers
            sample_videos.extend(sorted_videos[:sample_size//2])
            
            # Get mid-performers
            mid_start = len(sorted_videos) // 3
            mid_end = mid_start + sample_size//2
            sample_videos.extend(sorted_videos[mid_start:mid_end])
            
            # Get bottom performers for comparison
            sample_videos.extend(sorted_videos[-(sample_size//3):])
        else:
            sample_videos = channel_videos
        
        for video in sample_videos:
            try:
                analysis_result = self._analyze_single_video(video)
                if analysis_result:
                    results['analyzed_videos'].append(analysis_result)
                    
            except Exception as e:
                results['errors'].append({
                    'video_id': video.get('video_id', 'unknown'),
                    'error': str(e)
                })
        
        # Extract patterns from analyzed videos
        if results['analyzed_videos']:
            results['content_themes'] = self._extract_content_themes(results['analyzed_videos'])
            results['success_patterns'] = self._identify_success_patterns(results['analyzed_videos'])
            results['ai_insights'] = self._generate_ai_insights(results['analyzed_videos'])
        
        return results
    
    def _analyze_single_video(self, video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single video using the existing pipeline"""
        
        video_url = video.get('url')
        if not video_url:
            return None
        
        try:
            # Check if video already analyzed
            existing_video = get_video_by_url(video_url)
            
            if not existing_video:
                # Download video for analysis
                video_info = self.downloader.download_video(video_url)
                
                # Save to database
                video_id = save_video(
                    video_url,
                    video_info['title'],
                    video_info['duration'],
                    video_info['file_path'],
                    0  # Token count - will be calculated if needed
                )
                
                # Analyze first 2 minutes for content sample
                analysis = self.base_analyzer.analyze_interval(
                    video_info['file_path'], 
                    0, 
                    min(120, video_info['duration'])  # First 2 minutes or full video if shorter
                )
                
                if analysis:
                    analysis_id = save_analysis(video_id, 0, min(120, video_info['duration']), analysis)
                    
                    return {
                        'video_id': video_id,
                        'url': video_url,
                        'title': video_info['title'],
                        'duration': video_info['duration'],
                        'view_count': video.get('view_count', 0),
                        'like_count': video.get('like_count', 0),
                        'analysis': {
                            'topics': analysis.topics,
                            'summary': analysis.summary,
                            'key_points': analysis.key_points,
                            'entities': analysis.entities,
                            'sentiment': analysis.sentiment,
                            'confidence_score': analysis.confidence_score
                        }
                    }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing video {video_url}: {e}")
            return None
    
    def _extract_content_themes(self, analyzed_videos: List[Dict]) -> List[Dict[str, Any]]:
        """Extract common content themes from analyzed videos"""
        
        all_topics = []
        all_entities = []
        performance_data = []
        
        for video in analyzed_videos:
            analysis = video.get('analysis', {})
            all_topics.extend(analysis.get('topics', []))
            all_entities.extend(analysis.get('entities', []))
            
            performance_data.append({
                'topics': analysis.get('topics', []),
                'entities': analysis.get('entities', []),
                'view_count': video.get('view_count', 0),
                'sentiment': analysis.get('sentiment', 'neutral')
            })
        
        # Count topic frequency
        topic_counts = pd.Series(all_topics).value_counts()
        entity_counts = pd.Series(all_entities).value_counts()
        
        # Identify high-performing themes
        high_performing_themes = []
        for video_perf in performance_data:
            if video_perf['view_count'] > np.median([v['view_count'] for v in analyzed_videos]):
                high_performing_themes.extend(video_perf['topics'])
        
        high_perf_topic_counts = pd.Series(high_performing_themes).value_counts()
        
        return {
            'all_topics': topic_counts.head(10).to_dict(),
            'all_entities': entity_counts.head(10).to_dict(),
            'high_performing_topics': high_perf_topic_counts.head(5).to_dict(),
            'sentiment_distribution': pd.Series([v['sentiment'] for v in performance_data]).value_counts().to_dict()
        }
    
    def _identify_success_patterns(self, analyzed_videos: List[Dict]) -> Dict[str, Any]:
        """Identify patterns that correlate with success"""
        
        if not analyzed_videos:
            return {}
        
        # Create DataFrame for analysis
        pattern_data = []
        for video in analyzed_videos:
            analysis = video.get('analysis', {})
            pattern_data.append({
                'view_count': video.get('view_count', 0),
                'like_count': video.get('like_count', 0),
                'duration': video.get('duration', 0),
                'topic_count': len(analysis.get('topics', [])),
                'entity_count': len(analysis.get('entities', [])),
                'key_points_count': len(analysis.get('key_points', [])),
                'sentiment': analysis.get('sentiment', 'neutral'),
                'confidence_score': analysis.get('confidence_score', 0)
            })
        
        df = pd.DataFrame(pattern_data)
        
        # Calculate correlations with view count
        numeric_cols = ['duration', 'topic_count', 'entity_count', 'key_points_count', 'confidence_score']
        correlations = df[numeric_cols + ['view_count']].corr()['view_count'].drop('view_count')
        
        # Identify sentiment performance
        sentiment_performance = df.groupby('sentiment')['view_count'].mean().to_dict()
        
        return {
            'correlations': correlations.to_dict(),
            'sentiment_performance': sentiment_performance,
            'optimal_ranges': {
                'topic_count': df[df['view_count'] > df['view_count'].median()]['topic_count'].mean(),
                'duration': df[df['view_count'] > df['view_count'].median()]['duration'].mean(),
                'confidence_threshold': df[df['view_count'] > df['view_count'].median()]['confidence_score'].mean()
            }
        }
    
    def _generate_ai_insights(self, analyzed_videos: List[Dict]) -> List[str]:
        """Generate AI-powered insights from the analysis"""
        
        insights = []
        
        if not analyzed_videos:
            return ["No videos analyzed - unable to generate insights"]
        
        # Performance insights
        view_counts = [v.get('view_count', 0) for v in analyzed_videos]
        avg_views = sum(view_counts) / len(view_counts)
        
        # Content insights
        all_summaries = [v.get('analysis', {}).get('summary', '') for v in analyzed_videos]
        all_topics = []
        for v in analyzed_videos:
            all_topics.extend(v.get('analysis', {}).get('topics', []))
        
        common_topics = pd.Series(all_topics).value_counts().head(3).index.tolist()
        
        insights.extend([
            f"Average video performance: {avg_views:,.0f} views",
            f"Most successful content themes: {', '.join(common_topics)}",
            f"Analyzed {len(analyzed_videos)} videos with {len([v for v in analyzed_videos if v.get('analysis', {}).get('confidence_score', 0) > 0.7])} high-confidence analyses"
        ])
        
        # Performance patterns
        high_performers = [v for v in analyzed_videos if v.get('view_count', 0) > avg_views]
        if high_performers:
            high_perf_topics = []
            for v in high_performers:
                high_perf_topics.extend(v.get('analysis', {}).get('topics', []))
            
            if high_perf_topics:
                top_high_perf_topic = pd.Series(high_perf_topics).value_counts().index[0]
                insights.append(f"High-performing videos often feature: {top_high_perf_topic}")
        
        return insights

def integrate_with_strategy_app():
    """Integration function for the main strategy app"""
    return StrategyVideoAnalyzer()

# Example usage
if __name__ == "__main__":
    analyzer = StrategyVideoAnalyzer()
    
    # Test with sample video data
    sample_videos = [
        {
            'video_id': 'test123',
            'url': 'https://www.youtube.com/watch?v=test123',
            'view_count': 10000,
            'like_count': 500
        }
    ]
    
    results = analyzer.analyze_channel_videos(sample_videos, sample_size=1)
    print(json.dumps(results, indent=2))