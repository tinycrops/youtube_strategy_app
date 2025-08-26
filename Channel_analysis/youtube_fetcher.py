from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple, Callable

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import json
import re as _re
import xml.etree.ElementTree as ET

# ------------------------------
# Helpers
# ------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class TransientHTTPError(Exception):
    pass


class QuotaExceededError(Exception):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientHTTPError),
)
def _http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    resp = requests.get(url, params=params, headers=hdrs, timeout=30)
    if resp.status_code >= 500:
        raise TransientHTTPError(f"HTTP {resp.status_code} for {url}")
    if resp.status_code >= 400:
        # Surface the error payload to make debugging easier (quotaExceeded, keyInvalid, etc.)
        error_body: str
        try:
            # Prefer JSON if available
            error_json = resp.json()
            error_body = json.dumps(error_json, ensure_ascii=False)
        except Exception:
            # Fallback to raw text (truncate to avoid huge payloads)
            text = resp.text or ""
            error_body = text[:2000]
        # Redact API key from URL if present
        try:
            u = urlsplit(resp.url)
            q = dict(parse_qsl(u.query, keep_blank_values=True))
            if 'key' in q:
                q['key'] = 'REDACTED'
            safe_url = urlunsplit((u.scheme, u.netloc, u.path, urlencode(q), u.fragment))
        except Exception:
            safe_url = url
        message = f"HTTP {resp.status_code} for {safe_url} | body: {error_body}"
        raise requests.HTTPError(message, response=resp)
    # Success
    return resp.json()


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientHTTPError),
)
def _http_get_text(url: str, headers: Optional[dict] = None) -> str:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    resp = requests.get(url, headers=hdrs, timeout=30)
    if resp.status_code >= 500:
        raise TransientHTTPError(f"HTTP {resp.status_code} for {url}")
    resp.raise_for_status()
    return resp.text


# ------------------------------
# Public API
# ------------------------------

@dataclass
class VideoInfo:
    video_id: str
    title: str
    published_at: Optional[datetime]
    view_count: Optional[int]
    duration_seconds: Optional[int] = None
    like_count: Optional[int] = None
    top_comments: Optional[List[dict]] = None
    transcript_text: Optional[str] = None
    transcript_language: Optional[str] = None

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def normalize_channel_input(channel_input: str) -> str:
    """
    Accepts a channel handle like '@SleeplessHistorian', a channel page, or the '/videos' URL
    and returns a normalized '.../@handle/videos' style URL.
    """
    s = channel_input.strip()
    if not s:
        raise ValueError("Channel input is empty")
    if s.startswith("http://") or s.startswith("https://"):
        # Prefer the /videos tab to ensure we hit the creator page reliably
        if "/videos" in s:
            return s
        if s.endswith('/'):
            return s + "videos"
        return s + "/videos"
    # Assume it's a handle
    if s.startswith('@'):
        handle = s
    else:
        handle = '@' + s
    return f"https://www.youtube.com/{handle}/videos"


def resolve_channel_id_from_web(channel_url_or_videos_url: str) -> Optional[str]:
    """Resolve channelId (UC...) by scraping the channel page HTML."""
    html = _http_get_text(channel_url_or_videos_url)
    # Try several common patterns
    patterns = [
        r'"channelId"\s*:\s*"(UC[\w-]{20,})"',
        r'"externalId"\s*:\s*"(UC[\w-]{20,})"',
        r'"browseId"\s*:\s*"(UC[\w-]{20,})"',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None


def get_uploads_playlist_id(channel_id: str) -> str:
    if not channel_id.startswith('UC'):
        raise ValueError("channel_id must start with 'UC'")
    return 'UU' + channel_id[2:]


def _parse_iso8601_duration_to_seconds(duration: str) -> Optional[int]:
    """Parse ISO8601 duration (e.g., PT1H2M3S) to seconds."""
    if not duration:
        return None
    # Basic ISO8601 duration parser for YouTube style strings
    pattern = _re.compile(
        r"P(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?T?(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?"
    )
    m = pattern.fullmatch(duration)
    if not m:
        return None
    hours = int(m.group('h') or 0)
    minutes = int(m.group('m') or 0)
    seconds = int(m.group('s') or 0)
    return hours * 3600 + minutes * 60 + seconds


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type(Exception),
)
def _fetch_transcript_text(video_id: str, languages: List[str], max_chars: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    """Fetch transcript text for a video, returning (text, language)."""
    try:
        # Import inside to make dependency optional
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return None, None
    # Helper to join segments and enforce char limit
    def _segments_to_text(segments: List[dict]) -> str:
        parts = [seg.get('text') for seg in segments if seg.get('text')]
        text_local = ' '.join(parts)
        text_local = _re.sub(r"\s+", " ", text_local).strip()
        if max_chars is not None and isinstance(max_chars, int) and max_chars > 0 and len(text_local) > max_chars:
            text_local = text_local[:max_chars].rstrip() + '…'
        return text_local

    # 1) Try preferred languages
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages or ['en'])
        lang = segments[0].get('language') if segments else None
        text = _segments_to_text(segments)
        return (text or None), lang
    except Exception:
        pass

    # 2) Fallback to any available transcript (generated or manual)
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manually created transcripts first, then generated
        selected = None
        try:
            for l in languages or []:
                if listing.find_transcript([l]):
                    selected = listing.find_transcript([l])
                    break
        except Exception:
            selected = None
        if selected is None:
            # Choose the first transcript available
            for tr in listing:
                selected = tr
                break
        if selected is None:
            return None, None
        # Attempt translation to preferred language if not matching
        tr_obj = selected
        try:
            if languages and selected.language_code not in languages and selected.is_translatable:
                tr_obj = selected.translate(languages[0])
        except Exception:
            tr_obj = selected
        segments = tr_obj.fetch()
        text = _segments_to_text(segments)
        lang = getattr(tr_obj, 'language_code', None)
        return (text or None), lang
    except Exception:
        return None, None


def _extract_text_from_vtt(vtt_text: str, max_chars: Optional[int]) -> str:
    lines = []
    for line in vtt_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('WEBVTT'):
            continue
        # Skip cue numbers and time ranges
        if _re.match(r"^\d+$", s):
            continue
        if '-->' in s:
            continue
        lines.append(s)
    text = ' '.join(lines)
    text = _re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + '…'
    return text


def _extract_text_from_xml_captions(xml_text: str, max_chars: Optional[int]) -> str:
    try:
        root = ET.fromstring(xml_text)
        texts: List[str] = []
        for node in root.iter():
            if node.tag.lower().endswith('text') and (node.text or '').strip():
                texts.append(node.text.strip())
        text = ' '.join(texts)
        text = _re.sub(r"\s+", " ", text).strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + '…'
        return text
    except Exception:
        return ''


def _fetch_transcript_from_captions_map(captions_map: Optional[dict], languages: List[str], max_chars: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    if not captions_map or not isinstance(captions_map, dict):
        return None, None
    # Try preferred languages, then any English, then any available
    candidate_langs: List[str] = []
    candidate_langs.extend(languages or [])
    for en_key in ['en', 'en-US', 'en-GB']:
        if en_key not in candidate_langs:
            candidate_langs.append(en_key)
    candidate_langs.extend(list(captions_map.keys()))

    seen = set()
    for lang in candidate_langs:
        if lang in seen:
            continue
        seen.add(lang)
        tracks = captions_map.get(lang)
        if not tracks:
            continue
        # Prefer vtt track
        track = None
        for t in tracks:
            if t.get('ext') == 'vtt':
                track = t
                break
        if track is None:
            track = tracks[0]
        url = track.get('url')
        if not url:
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            content = resp.text
            ext = track.get('ext') or ''
            if ext.lower() == 'vtt':
                text = _extract_text_from_vtt(content, max_chars)
            else:
                # try XML
                text = _extract_text_from_xml_captions(content, max_chars)
                if not text:
                    # last resort: treat as plain text
                    text = _extract_text_from_vtt(content, max_chars)
            if text:
                return text, lang
        except Exception:
            continue
    return None, None


def _fetch_top_comments_via_api(video_id: str, api_key: str, limit: int = 20) -> List[dict]:
    """Fetch up to `limit` top comments (by relevance) for a video via Data API."""
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': min(max(limit, 1), 100),
        'order': 'relevance',
        'textFormat': 'plainText',
        'key': api_key,
    }
    try:
        data = _http_get_json(base_url, params=params)
    except requests.HTTPError as e:
        # Gracefully degrade on 4xx (e.g., 403 forbidden, comments disabled, quota issues)
        return []
    except Exception:
        return []
    comments: List[dict] = []
    for item in data.get('items', []):
        top = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
        if not top:
            continue
        comments.append({
            'author': top.get('authorDisplayName'),
            'text': top.get('textDisplay') or top.get('textOriginal'),
            'like_count': top.get('likeCount'),
            'published_at': top.get('publishedAt'),
            'updated_at': top.get('updatedAt'),
        })
        if len(comments) >= limit:
            break
    return comments


# ------------------------------
# YouTube Data API (fast)
# ------------------------------

def fetch_videos_via_api(channel_input: str, api_key: str, *, include_comments: bool = True, comment_limit: int = 20, include_transcript: bool = False, transcript_languages: Optional[List[str]] = None, transcript_max_chars: Optional[int] = None, max_videos: Optional[int] = None, existing_video_ids: Optional[Iterable[str]] = None, progress_callback: Optional[Callable[[List[VideoInfo]], None]] = None) -> List[VideoInfo]:
    """
    Use YouTube Data API v3 to list all uploaded videos and statistics.
    Only requires API key (no OAuth).
    """
    videos_url = normalize_channel_input(channel_input)
    channel_id = resolve_channel_id_from_web(videos_url)
    if not channel_id:
        raise RuntimeError("Failed to resolve channelId from the provided channel")

    uploads_id = get_uploads_playlist_id(channel_id)

    # 1) Enumerate all video IDs from uploads playlist
    base_playlist_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    params = {
        'part': 'contentDetails',
        'playlistId': uploads_id,
        'maxResults': 50,
        'key': api_key,
    }

    video_ids: List[str] = []
    page_token: Optional[str] = None
    quota_hit = False
    while True:
        if page_token:
            params['pageToken'] = page_token
        else:
            params.pop('pageToken', None)
        try:
            data = _http_get_json(base_playlist_url, params=params)
        except requests.HTTPError as e:
            # If quota is exceeded, return what we have so far
            msg = str(e).lower()
            if 'quota' in msg and 'exceed' in msg:
                quota_hit = True
                break
            raise
        for item in data.get('items', []):
            vid = item.get('contentDetails', {}).get('videoId')
            if vid:
                video_ids.append(vid)
                if max_videos is not None and len(video_ids) >= max_videos:
                    page_token = None
                    break
        page_token = data.get('nextPageToken')
        if not page_token:
            break

    # 2) Fetch statistics in batches of up to 50
    base_videos_url = "https://www.googleapis.com/youtube/v3/videos"
    videos: List[VideoInfo] = []
    skip_ids = set(existing_video_ids or [])
    for i in range(0, len(video_ids), 50):
        batch_all = video_ids[i:i+50]
        # Skip already-saved videos when resuming
        batch = [vid for vid in batch_all if vid not in skip_ids]
        if not batch:
            continue
        vparams = {
            'part': 'snippet,statistics,contentDetails',
            'id': ','.join(batch),
            'maxResults': 50,
            'key': api_key,
        }
        try:
            vdata = _http_get_json(base_videos_url, params=vparams)
        except requests.HTTPError as e:
            msg = str(e).lower()
            if 'quota' in msg and 'exceed' in msg:
                quota_hit = True
                break
            raise
        new_videos_batch: List[VideoInfo] = []
        for item in vdata.get('items', []):
            vid = item.get('id')
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            cdetails = item.get('contentDetails', {})
            title = snippet.get('title') or ""
            published_at_str = snippet.get('publishedAt')
            published_at = None
            if published_at_str:
                try:
                    published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                except Exception:
                    published_at = None
            view_count = None
            vc = stats.get('viewCount')
            if vc is not None:
                try:
                    view_count = int(vc)
                except Exception:
                    view_count = None
            like_count = None
            lc = stats.get('likeCount')
            if lc is not None:
                try:
                    like_count = int(lc)
                except Exception:
                    like_count = None
            duration_seconds = _parse_iso8601_duration_to_seconds(cdetails.get('duration')) if cdetails else None
            # Only fetch comments for new videos
            comments = _fetch_top_comments_via_api(vid, api_key, limit=comment_limit) if (include_comments and vid and vid not in skip_ids) else None
            transcript_text, transcript_language = (None, None)
            if include_transcript and vid and vid not in skip_ids:
                transcript_text, transcript_language = _fetch_transcript_text(vid, transcript_languages or ['en', 'en-US', 'en-GB'], transcript_max_chars)
            vi = VideoInfo(
                video_id=vid,
                title=title,
                published_at=published_at,
                view_count=view_count,
                duration_seconds=duration_seconds,
                like_count=like_count,
                top_comments=comments,
                transcript_text=transcript_text,
                transcript_language=transcript_language,
            )
            videos.append(vi)
            new_videos_batch.append(vi)
        if progress_callback and new_videos_batch:
            try:
                progress_callback(new_videos_batch)
            except Exception:
                pass
        if quota_hit:
            break
    if quota_hit:
        # Signal to caller so it can stop further channels. Partial results may have been saved via callback.
        raise QuotaExceededError("YouTube Data API quotaExceeded during fetch; partial results saved if progress_callback was provided.")
    return videos


# ------------------------------
# yt-dlp fallback (no API key)
# ------------------------------

def fetch_videos_via_ytdlp(channel_input: str, max_videos: Optional[int] = None, *, include_comments: bool = True, comment_limit: int = 20, include_transcript: bool = False, transcript_languages: Optional[List[str]] = None, transcript_max_chars: Optional[int] = None) -> List[VideoInfo]:
    """
    Use yt-dlp to extract the channel videos and per-video metadata.
    Note: This can be slow for large channels because it fetches each video page.
    """
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        raise RuntimeError("yt-dlp is not installed. Install it via pip.") from e

    url = normalize_channel_input(channel_input)

    # First, extract the entries (video ids) from the videos tab.
    # We'll avoid flat extraction so titles and basic fields come through, but to get
    # consistent view counts we may still need to pull per-video; yt-dlp generally does it.
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': False,  # allows per-video extraction for more fields
        'nocheckcertificate': True,
        'consoletitle': False,
        'noprogress': True,
    }
    if include_comments:
        # Ask yt-dlp to retrieve comments with a limit and sort by top
        ydl_opts['extractor_args'] = {
            'youtube': {
                'max_comments': [str(max(1, comment_limit))],
                'comment_sort': ['top'],
            }
        }

    videos: List[VideoInfo] = []
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # Sometimes the channel page returns a playlist-like with 'entries'
        entries = info.get('entries', []) if isinstance(info, dict) else []
        count = 0
        for entry in entries:
            if max_videos is not None and count >= max_videos:
                break
            # Entries may be partial; if a URL is present, yt-dlp can expand
            if entry.get('ie_key') == 'Youtube':
                # already a full video dict
                pass
            elif entry.get('url'):
                sub = ydl.extract_info(entry['url'], download=False)
                entry = sub if isinstance(sub, dict) else entry

            vid = entry.get('id')
            title = entry.get('title') or ""
            # yt-dlp names this 'view_count'
            vc = entry.get('view_count')
            view_count = int(vc) if isinstance(vc, int) else (int(vc) if isinstance(vc, str) and vc.isdigit() else None)
            like_count = entry.get('like_count')
            if like_count is not None and not isinstance(like_count, int):
                try:
                    like_count = int(like_count)
                except Exception:
                    like_count = None
            # published date might be 'upload_date' as YYYYMMDD
            published_at = None
            if entry.get('upload_date'):
                try:
                    published_at = datetime.strptime(entry['upload_date'], '%Y%m%d').replace(tzinfo=timezone.utc)
                except Exception:
                    published_at = None
            elif entry.get('release_timestamp'):
                try:
                    published_at = datetime.fromtimestamp(entry['release_timestamp'], tz=timezone.utc)
                except Exception:
                    published_at = None
            duration_seconds = entry.get('duration') if isinstance(entry.get('duration'), int) else None
            comments: Optional[List[dict]] = None
            if include_comments and isinstance(entry.get('comments'), list):
                # Normalize and keep top by like_count
                normalized: List[dict] = []
                for c in entry['comments']:
                    try:
                        normalized.append({
                            'author': c.get('author'),
                            'text': c.get('text'),
                            'like_count': int(c.get('like_count')) if c.get('like_count') is not None else None,
                            'timestamp': c.get('timestamp'),
                        })
                    except Exception:
                        continue
                normalized.sort(key=lambda x: (x.get('like_count') or 0), reverse=True)
                comments = normalized[:comment_limit]
            transcript_text, transcript_language = (None, None)
            if include_transcript and vid:
                # Prefer direct captions URLs (faster, avoids API bans), fallback to transcript API
                transcript_text, transcript_language = _fetch_transcript_from_captions_map(entry.get('automatic_captions') or entry.get('subtitles'), transcript_languages or ['en','en-US','en-GB'], transcript_max_chars)
                if not transcript_text:
                    transcript_text, transcript_language = _fetch_transcript_text(vid, transcript_languages or ['en', 'en-US', 'en-GB'], transcript_max_chars)

            if vid:
                videos.append(VideoInfo(
                    video_id=vid,
                    title=title,
                    published_at=published_at,
                    view_count=view_count,
                    duration_seconds=duration_seconds,
                    like_count=like_count,
                    top_comments=comments,
                    transcript_text=transcript_text,
                    transcript_language=transcript_language,
                ))
                count += 1
    return videos


# ------------------------------
# Public facade
# ------------------------------

def videos_to_dataframe(videos: List[VideoInfo]) -> pd.DataFrame:
    rows = []
    for v in videos:
        rows.append({
            'video_id': v.video_id,
            'title': v.title,
            'published_at': v.published_at,
            'view_count': v.view_count,
            'duration_seconds': v.duration_seconds,
            'like_count': v.like_count,
            'top_comments': json.dumps(v.top_comments, ensure_ascii=False) if v.top_comments else None,
            'url': v.watch_url,
            'transcript_text': v.transcript_text,
            'transcript_language': v.transcript_language,
        })
    df = pd.DataFrame(rows)
    if not df.empty and 'published_at' in df.columns:
        df = df.sort_values(by='published_at', ascending=False, na_position='last').reset_index(drop=True)
    return df


def fetch_channel_videos(channel_input: str, method: str = 'api', api_key: Optional[str] = None, max_videos: Optional[int] = None, *, include_comments: bool = True, comment_limit: int = 20, include_transcript: bool = False, transcript_languages: Optional[List[str]] = None, transcript_max_chars: Optional[int] = None, existing_video_ids: Optional[Iterable[str]] = None, progress_callback: Optional[Callable[[List[VideoInfo]], None]] = None) -> pd.DataFrame:
    """
    method: 'api' or 'ytdlp'
    - api: requires api_key
    - ytdlp: no key, slower
    """
    if method == 'api':
        key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not key:
            raise ValueError("API method selected but no API key provided. Set YOUTUBE_API_KEY or pass api_key.")
        videos = fetch_videos_via_api(
            channel_input,
            key,
            include_comments=include_comments,
            comment_limit=comment_limit,
            include_transcript=include_transcript,
            transcript_languages=transcript_languages,
            transcript_max_chars=transcript_max_chars,
            max_videos=max_videos,
            existing_video_ids=existing_video_ids,
            progress_callback=progress_callback,
        )
    elif method == 'ytdlp':
        videos = fetch_videos_via_ytdlp(channel_input, max_videos=max_videos, include_comments=include_comments, comment_limit=comment_limit, include_transcript=include_transcript, transcript_languages=transcript_languages, transcript_max_chars=transcript_max_chars)
    else:
        raise ValueError("method must be 'api' or 'ytdlp'")

    return videos_to_dataframe(videos)
