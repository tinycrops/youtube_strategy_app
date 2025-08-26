#!/usr/bin/env python3
"""
Advanced Thumbnail Analysis for YouTube Strategy
AI-powered thumbnail analysis using computer vision and machine learning
"""

import requests
import cv2
import numpy as np
from PIL import Image, ImageStat, ImageFilter
import io
import base64
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import colorsys
import google.genai as genai
import os
from pathlib import Path
import tempfile

class ThumbnailAnalyzer:
    """Advanced thumbnail analysis with AI insights"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        if gemini_api_key:
            self.client = genai.Client(api_key=gemini_api_key)
            self.model = 'gemini-1.5-flash'
        else:
            self.model = None
    
    def analyze_thumbnail_batch(self, video_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple thumbnails for patterns and effectiveness"""
        
        results = {
            'thumbnail_analyses': [],
            'color_patterns': {},
            'composition_insights': {},
            'performance_correlations': {},
            'ai_recommendations': [],
            'failed_analyses': []
        }
        
        valid_thumbnails = []
        
        # Analyze individual thumbnails
        for video in video_data:
            thumbnail_url = self._extract_thumbnail_url(video)
            if thumbnail_url:
                try:
                    analysis = self.analyze_single_thumbnail(
                        thumbnail_url, 
                        video.get('view_count', 0),
                        video.get('title', ''),
                        video.get('video_id', '')
                    )
                    if analysis:
                        results['thumbnail_analyses'].append(analysis)
                        valid_thumbnails.append(video)
                except Exception as e:
                    results['failed_analyses'].append({
                        'video_id': video.get('video_id', 'unknown'),
                        'error': str(e)
                    })
        
        # Perform batch analyses if we have valid thumbnails
        if results['thumbnail_analyses']:
            results['color_patterns'] = self._analyze_color_patterns(results['thumbnail_analyses'])
            results['composition_insights'] = self._analyze_composition_patterns(results['thumbnail_analyses'])
            results['performance_correlations'] = self._correlate_with_performance(results['thumbnail_analyses'], valid_thumbnails)
            
            if hasattr(self, 'client') and self.client:
                results['ai_recommendations'] = self._generate_ai_recommendations(results)
        
        return results
    
    def analyze_single_thumbnail(self, thumbnail_url: str, view_count: int, title: str, video_id: str) -> Optional[Dict[str, Any]]:
        """Analyze a single thumbnail for design elements"""
        
        try:
            # Download thumbnail
            response = requests.get(thumbnail_url, timeout=10)
            if response.status_code != 200:
                return None
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            # Basic image properties
            width, height = img.size
            aspect_ratio = width / height
            
            # Color analysis
            color_analysis = self._analyze_colors(img, img_array)
            
            # Composition analysis
            composition_analysis = self._analyze_composition(img_array)
            
            # Text detection (basic)
            text_analysis = self._analyze_text_presence(img_array)
            
            # Face detection
            face_analysis = self._detect_faces(img_array)
            
            # Visual complexity
            complexity = self._calculate_visual_complexity(img_array)
            
            return {
                'video_id': video_id,
                'title': title,
                'view_count': view_count,
                'thumbnail_url': thumbnail_url,
                'dimensions': {'width': width, 'height': height},
                'aspect_ratio': aspect_ratio,
                'color_analysis': color_analysis,
                'composition': composition_analysis,
                'text_analysis': text_analysis,
                'face_analysis': face_analysis,
                'visual_complexity': complexity
            }
            
        except Exception as e:
            print(f"Error analyzing thumbnail {thumbnail_url}: {e}")
            return None
    
    def _extract_thumbnail_url(self, video_data: Dict[str, Any]) -> Optional[str]:
        """Extract thumbnail URL from video data"""
        
        # Try different possible thumbnail URL formats
        video_id = video_data.get('video_id', '')
        
        if 'thumbnail' in video_data and video_data['thumbnail']:
            return video_data['thumbnail']
        
        # Generate YouTube thumbnail URL
        if video_id:
            # Try high quality first, then fallback
            thumbnail_urls = [
                f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            ]
            
            for url in thumbnail_urls:
                try:
                    response = requests.head(url, timeout=5)
                    if response.status_code == 200:
                        return url
                except:
                    continue
        
        return None
    
    def _analyze_colors(self, img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color composition and schemes"""
        
        # Dominant colors using K-means
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        dominant_colors = kmeans.cluster_centers_.astype(int)
        color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
        
        # Color temperature (warm vs cool)
        avg_color = np.mean(pixels, axis=0)
        color_temp = self._calculate_color_temperature(avg_color)
        
        # Brightness and contrast
        brightness = ImageStat.Stat(img).mean
        contrast = ImageStat.Stat(img).stddev
        
        # Saturation analysis
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv_img[:, :, 1])
        
        return {
            'dominant_colors': dominant_colors.tolist(),
            'color_percentages': color_percentages.tolist(),
            'color_temperature': color_temp,
            'brightness': brightness,
            'contrast': contrast,
            'average_saturation': float(avg_saturation),
            'color_scheme': self._classify_color_scheme(dominant_colors)
        }
    
    def _analyze_composition(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze composition elements like rule of thirds, symmetry"""
        
        height, width = img_array.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Rule of thirds analysis
        thirds_analysis = self._analyze_rule_of_thirds(gray)
        
        # Symmetry analysis
        symmetry_score = self._calculate_symmetry(gray)
        
        # Focus analysis (blur detection)
        focus_score = self._calculate_focus_score(gray)
        
        return {
            'edge_density': float(edge_density),
            'rule_of_thirds': thirds_analysis,
            'symmetry_score': float(symmetry_score),
            'focus_score': float(focus_score)
        }
    
    def _analyze_text_presence(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Basic text detection and analysis"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological operations to detect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that could be text
        text_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Basic text-like criteria
            if 0.1 < aspect_ratio < 10 and 50 < area < 10000:
                text_contours.append(contour)
        
        # Estimate text coverage
        total_area = img_array.shape[0] * img_array.shape[1]
        text_area = sum(cv2.contourArea(c) for c in text_contours)
        text_coverage = text_area / total_area
        
        return {
            'text_regions_detected': len(text_contours),
            'estimated_text_coverage': float(text_coverage),
            'has_text': len(text_contours) > 0
        }
    
    def _detect_faces(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Basic face detection"""
        
        try:
            # Use OpenCV's cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_info = []
            for (x, y, w, h) in faces:
                face_info.append({
                    'position': [int(x), int(y)],
                    'size': [int(w), int(h)],
                    'area_percentage': float((w * h) / (img_array.shape[0] * img_array.shape[1]))
                })
            
            return {
                'faces_detected': len(faces),
                'face_details': face_info,
                'total_face_coverage': sum(f['area_percentage'] for f in face_info)
            }
            
        except Exception as e:
            return {
                'faces_detected': 0,
                'face_details': [],
                'total_face_coverage': 0.0,
                'error': str(e)
            }
    
    def _calculate_visual_complexity(self, img_array: np.ndarray) -> float:
        """Calculate visual complexity using edge density and color variation"""
        
        # Edge complexity
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_complexity = np.sum(edges > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # Color complexity (standard deviation of colors)
        color_std = np.std(img_array.reshape(-1, 3), axis=0).mean()
        
        # Combine metrics (normalized)
        complexity_score = (edge_complexity * 100 + color_std / 255) / 2
        
        return float(complexity_score)
    
    def _calculate_color_temperature(self, rgb_color: np.ndarray) -> str:
        """Calculate if color is warm or cool"""
        r, g, b = rgb_color
        
        # Convert to HSV for better analysis
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h_degrees = h * 360
        
        if 0 <= h_degrees <= 60 or 300 <= h_degrees <= 360:
            return "warm"
        elif 120 <= h_degrees <= 240:
            return "cool"
        else:
            return "neutral"
    
    def _classify_color_scheme(self, dominant_colors: np.ndarray) -> str:
        """Classify the color scheme type"""
        
        if len(dominant_colors) < 2:
            return "monochromatic"
        
        # Convert to HSV for hue analysis
        hsv_colors = []
        for color in dominant_colors:
            r, g, b = color / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append([h * 360, s, v])
        
        hsv_colors = np.array(hsv_colors)
        hue_std = np.std(hsv_colors[:, 0])
        
        if hue_std < 30:
            return "monochromatic"
        elif hue_std < 60:
            return "analogous"
        elif hue_std > 120:
            return "complementary"
        else:
            return "triadic"
    
    def _analyze_rule_of_thirds(self, gray_img: np.ndarray) -> Dict[str, Any]:
        """Analyze adherence to rule of thirds"""
        
        height, width = gray_img.shape
        
        # Define thirds lines
        v_third1, v_third2 = width // 3, 2 * width // 3
        h_third1, h_third2 = height // 3, 2 * height // 3
        
        # Calculate interest points at intersections
        intersections = [
            (v_third1, h_third1), (v_third1, h_third2),
            (v_third2, h_third1), (v_third2, h_third2)
        ]
        
        # Calculate edge density near intersections
        edge_density_at_intersections = 0
        for x, y in intersections:
            region = gray_img[max(0, y-20):min(height, y+20), max(0, x-20):min(width, x+20)]
            edges = cv2.Canny(region, 50, 150)
            edge_density_at_intersections += np.sum(edges > 0) / (40 * 40)
        
        return {
            'intersection_activity': float(edge_density_at_intersections / 4),
            'thirds_adherence_score': min(1.0, edge_density_at_intersections / 400)
        }
    
    def _calculate_symmetry(self, gray_img: np.ndarray) -> float:
        """Calculate symmetry score"""
        
        height, width = gray_img.shape
        
        # Vertical symmetry
        left_half = gray_img[:, :width//2]
        right_half = np.fliplr(gray_img[:, width//2:])
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate similarity
        diff = np.abs(left_half.astype(float) - right_half.astype(float))
        symmetry_score = 1 - (np.mean(diff) / 255)
        
        return max(0, symmetry_score)
    
    def _calculate_focus_score(self, gray_img: np.ndarray) -> float:
        """Calculate focus/sharpness score using Laplacian variance"""
        
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        focus_score = laplacian.var()
        
        # Normalize to 0-1 range (typical values range from 0-2000)
        return min(1.0, focus_score / 1000.0)
    
    def _analyze_color_patterns(self, thumbnail_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze color patterns across all thumbnails"""
        
        color_schemes = [t['color_analysis']['color_scheme'] for t in thumbnail_analyses]
        color_temps = [t['color_analysis']['color_temperature'] for t in thumbnail_analyses]
        brightness_values = [np.mean(t['color_analysis']['brightness']) for t in thumbnail_analyses]
        saturation_values = [t['color_analysis']['average_saturation'] for t in thumbnail_analyses]
        
        return {
            'color_scheme_distribution': pd.Series(color_schemes).value_counts().to_dict(),
            'color_temperature_distribution': pd.Series(color_temps).value_counts().to_dict(),
            'average_brightness': np.mean(brightness_values),
            'average_saturation': np.mean(saturation_values),
            'brightness_range': [min(brightness_values), max(brightness_values)],
            'saturation_range': [min(saturation_values), max(saturation_values)]
        }
    
    def _analyze_composition_patterns(self, thumbnail_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze composition patterns across thumbnails"""
        
        face_presence = [t['face_analysis']['faces_detected'] > 0 for t in thumbnail_analyses]
        text_presence = [t['text_analysis']['has_text'] for t in thumbnail_analyses]
        complexity_scores = [t['visual_complexity'] for t in thumbnail_analyses]
        focus_scores = [t['composition']['focus_score'] for t in thumbnail_analyses]
        
        return {
            'face_presence_rate': sum(face_presence) / len(face_presence),
            'text_presence_rate': sum(text_presence) / len(text_presence),
            'average_complexity': np.mean(complexity_scores),
            'average_focus': np.mean(focus_scores),
            'complexity_distribution': {
                'low': sum(1 for s in complexity_scores if s < 0.3),
                'medium': sum(1 for s in complexity_scores if 0.3 <= s < 0.7),
                'high': sum(1 for s in complexity_scores if s >= 0.7)
            }
        }
    
    def _correlate_with_performance(self, thumbnail_analyses: List[Dict], video_data: List[Dict]) -> Dict[str, Any]:
        """Correlate thumbnail features with video performance"""
        
        if not thumbnail_analyses:
            return {}
        
        # Create DataFrame for correlation analysis
        data = []
        for analysis in thumbnail_analyses:
            data.append({
                'view_count': analysis['view_count'],
                'brightness': np.mean(analysis['color_analysis']['brightness']),
                'saturation': analysis['color_analysis']['average_saturation'],
                'complexity': analysis['visual_complexity'],
                'has_faces': analysis['face_analysis']['faces_detected'] > 0,
                'has_text': analysis['text_analysis']['has_text'],
                'focus_score': analysis['composition']['focus_score'],
                'face_coverage': analysis['face_analysis']['total_face_coverage']
            })
        
        df = pd.DataFrame(data)
        
        # Calculate correlations with view count
        numeric_features = ['brightness', 'saturation', 'complexity', 'focus_score', 'face_coverage']
        correlations = df[numeric_features + ['view_count']].corr()['view_count'].drop('view_count')
        
        # Performance by categorical features
        performance_by_faces = df.groupby('has_faces')['view_count'].mean().to_dict()
        performance_by_text = df.groupby('has_text')['view_count'].mean().to_dict()
        
        return {
            'feature_correlations': correlations.to_dict(),
            'performance_by_face_presence': performance_by_faces,
            'performance_by_text_presence': performance_by_text,
            'high_performers_analysis': self._analyze_high_performers(df)
        }
    
    def _analyze_high_performers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of high-performing thumbnails"""
        
        if len(df) < 3:
            return {}
        
        # Define high performers as top 33%
        threshold = df['view_count'].quantile(0.67)
        high_performers = df[df['view_count'] >= threshold]
        
        if len(high_performers) == 0:
            return {}
        
        return {
            'avg_brightness': high_performers['brightness'].mean(),
            'avg_saturation': high_performers['saturation'].mean(),
            'avg_complexity': high_performers['complexity'].mean(),
            'face_presence_rate': high_performers['has_faces'].mean(),
            'text_presence_rate': high_performers['has_text'].mean(),
            'avg_focus_score': high_performers['focus_score'].mean()
        }
    
    def _generate_ai_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate AI-powered thumbnail recommendations"""
        
        if not hasattr(self, 'client') or not self.client:
            return self._generate_basic_recommendations(analysis_results)
        
        try:
            # Prepare analysis summary for AI
            summary = {
                'color_patterns': analysis_results.get('color_patterns', {}),
                'composition_insights': analysis_results.get('composition_insights', {}),
                'performance_correlations': analysis_results.get('performance_correlations', {})
            }
            
            prompt = f"""
            Based on this YouTube thumbnail analysis data, provide 5-7 specific, actionable recommendations for creating high-performing thumbnails:

            Analysis Summary:
            {json.dumps(summary, indent=2)}

            Focus on:
            1. Color choices and schemes
            2. Composition and visual elements
            3. Text and face usage
            4. Overall design strategy
            5. Performance optimization tips

            Provide concrete, implementable advice for content creators.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            recommendations = response.text.split('\n')
            
            # Clean and filter recommendations
            return [rec.strip() for rec in recommendations if rec.strip() and len(rec.strip()) > 20]
            
        except Exception as e:
            return self._generate_basic_recommendations(analysis_results)
    
    def _generate_basic_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations without AI"""
        
        recommendations = [
            "Use high contrast colors to make thumbnails stand out",
            "Include faces when possible - they increase click-through rates",
            "Add clear, readable text to communicate video content",
            "Maintain visual consistency across your thumbnail style",
            "Use bright, saturated colors to capture attention",
            "Follow the rule of thirds for better composition",
            "Ensure thumbnails are clear and focused, not blurry"
        ]
        
        # Customize based on analysis if available
        color_patterns = analysis_results.get('color_patterns', {})
        if color_patterns.get('average_brightness', 0) < 100:
            recommendations.append("Consider using brighter colors to improve visibility")
        
        composition = analysis_results.get('composition_insights', {})
        if composition.get('face_presence_rate', 0) < 0.3:
            recommendations.append("Consider including faces in more thumbnails")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    analyzer = ThumbnailAnalyzer()
    
    sample_videos = [
        {
            'video_id': 'dQw4w9WgXcQ',
            'view_count': 1000000,
            'title': 'Sample Video Title'
        }
    ]
    
    results = analyzer.analyze_thumbnail_batch(sample_videos)
    print(json.dumps(results, indent=2, default=str))