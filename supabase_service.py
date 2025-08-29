import os
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

_SUPABASE_URL_DEFAULT = "https://zycaxecnoszbcwrfeokn.supabase.co"
_SUPABASE_ANON_DEFAULT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5Y2F4ZWNub3N6YmN3cmZlb2tuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyMzIyNTUsImV4cCI6MjA3MTgwODI1NX0.q0FUC7vRHrQhReHBZsnFdeMez3Mg3tVP3j_qQzQEv14"
)


def _get_creds() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", _SUPABASE_URL_DEFAULT)
    key = os.getenv("SUPABASE_ANON_KEY", _SUPABASE_ANON_DEFAULT)
    return url, key


def get_client():
    try:
        from supabase import create_client
    except Exception as e:
        raise RuntimeError("Supabase client is not installed. Add 'supabase>=2.4.0' to requirements.") from e
    url, key = _get_creds()
    return create_client(url, key)


def _ensure_json_serializable(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serializable: List[Dict[str, Any]] = []
    for row in records:
        fixed: Dict[str, Any] = {}
        for k, v in row.items():
            # Normalize missing values
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)) or (hasattr(pd, 'isna') and pd.isna(v)):
                    fixed[k] = None
                    continue
            except Exception:
                pass

            # Datetime-like
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime().isoformat()
            elif isinstance(v, (datetime, date)):
                try:
                    v = v.isoformat()
                except Exception:
                    v = str(v)
            # Timedelta
            elif isinstance(v, getattr(pd, 'Timedelta', tuple())):
                try:
                    v = v.isoformat()
                except Exception:
                    v = str(v)
            # Numpy scalar types
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            # Bytes
            elif isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode('utf-8', errors='ignore')
                except Exception:
                    v = str(v)
            # Leave lists/dicts/strings as-is
            fixed[k] = v
        serializable.append(fixed)
    return serializable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_channel_analytics(channel_key: str, df: pd.DataFrame) -> None:
    client = get_client()
    data_records: List[Dict[str, Any]] = []
    try:
        data_records = df.to_dict(orient='records') if df is not None else []
    except Exception:
        data_records = []
    data_records = _ensure_json_serializable(data_records)
    payload = {
        "channel_key": channel_key,
        "data": data_records,
        "updated_at": _now_iso(),
    }
    try:
        client.table("channel_analytics").upsert(payload, on_conflict="channel_key").execute()
    except Exception:
        # Graceful no-op if table doesn't exist yet
        pass


def get_channel_analytics(channel_key: str) -> Optional[pd.DataFrame]:
    client = get_client()
    try:
        res = client.table("channel_analytics").select("data").eq("channel_key", channel_key).limit(1).execute()
        items = getattr(res, "data", None) or []
        if not items:
            return None
        records = items[0].get("data") or []
        if not records:
            return None
        try:
            return pd.DataFrame(records)
        except Exception:
            return None
    except Exception:
        return None


def list_saved_channel_analytics(limit: int = 50) -> Dict[str, pd.DataFrame]:
    """Return a mapping of channel_key -> DataFrame of saved analytics."""
    client = get_client()
    out: Dict[str, pd.DataFrame] = {}
    try:
        res = (
            client.table("channel_analytics")
            .select("channel_key, data, updated_at")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        items = getattr(res, "data", None) or []
        for row in items:
            key = row.get("channel_key")
            records = row.get("data") or []
            if key and isinstance(records, list) and records:
                try:
                    out[key] = pd.DataFrame(records)
                except Exception:
                    continue
        return out
    except Exception:
        return out


def upsert_strategy(channel_key: str, strategy: Dict[str, Any]) -> None:
    client = get_client()
    payload = {
        "channel_key": channel_key,
        "strategy": strategy or {},
        "updated_at": _now_iso(),
    }
    try:
        client.table("channel_strategies").upsert(payload, on_conflict="channel_key").execute()
    except Exception:
        pass


def get_strategy(channel_key: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    try:
        res = client.table("channel_strategies").select("strategy").eq("channel_key", channel_key).limit(1).execute()
        items = getattr(res, "data", None) or []
        if not items:
            return None
        strat = items[0].get("strategy")
        if isinstance(strat, dict):
            return strat
        return None
    except Exception:
        return None


# ===============
# Creator Persona
# ===============

def upsert_creator_persona(persona_key: str, persona: Dict[str, Any]) -> None:
    """Create or update a creator persona document.

    Expects a Supabase table `creator_personas` with columns:
      - persona_key (text, primary key)
      - persona (jsonb)
      - updated_at (timestamptz)
    """
    client = get_client()
    payload = {
        "persona_key": persona_key,
        "persona": persona or {},
        "updated_at": _now_iso(),
    }
    try:
        client.table("creator_personas").upsert(payload, on_conflict="persona_key").execute()
    except Exception:
        # Gracefully ignore if table is missing
        pass


def get_creator_persona(persona_key: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    try:
        res = (
            client.table("creator_personas")
            .select("persona")
            .eq("persona_key", persona_key)
            .limit(1)
            .execute()
        )
        items = getattr(res, "data", None) or []
        if not items:
            return None
        val = items[0].get("persona")
        return val if isinstance(val, dict) else None
    except Exception:
        return None


# =================
# Content Journal
# =================

def add_journal_entry(entry: Dict[str, Any]) -> None:
    """Insert a content journal entry.

    Expected table `content_journal` columns:
      - id (uuid default gen_random_uuid())
      - created_at (timestamptz)
      - updated_at (timestamptz)
      - entry (jsonb)
    """
    client = get_client()
    payload = {
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "entry": entry or {},
    }
    try:
        client.table("content_journal").insert(payload).execute()
    except Exception:
        pass


def list_journal_entries(limit: int = 50) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        res = (
            client.table("content_journal")
            .select("entry, created_at, updated_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        items = getattr(res, "data", None) or []
        out: List[Dict[str, Any]] = []
        for row in items:
            ent = row.get("entry") or {}
            if isinstance(ent, dict):
                ent["created_at"] = row.get("created_at")
                ent["updated_at"] = row.get("updated_at")
                out.append(ent)
        return out
    except Exception:
        return []


# =====================
# Preference Learning
# =====================

def record_preference_event(event: Dict[str, Any]) -> None:
    """Record a preference/feedback event.

    Expected table `preference_events` columns:
      - id (uuid)
      - created_at (timestamptz)
      - event (jsonb)
    """
    client = get_client()
    payload = {
        "created_at": _now_iso(),
        "event": event or {},
    }
    try:
        client.table("preference_events").insert(payload).execute()
    except Exception:
        pass


def list_preference_events(limit: int = 100) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        res = (
            client.table("preference_events")
            .select("event, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        items = getattr(res, "data", None) or []
        out: List[Dict[str, Any]] = []
        for row in items:
            ev = row.get("event") or {}
            if isinstance(ev, dict):
                ev["created_at"] = row.get("created_at")
                out.append(ev)
        return out
    except Exception:
        return []


def save_mvp_video_kit(channel_key: str, persona_key: str, kit: Dict[str, Any]) -> None:
    client = get_client()
    payload = {
        "channel_key": channel_key,
        "persona_key": persona_key,
        "kit": kit or {},
        "created_at": _now_iso(),
    }
    try:
        client.table("mvp_video_kits").insert(payload).execute()
    except Exception:
        pass


# Fetch latest saved MVP kit for a channel/persona
def get_latest_mvp_video_kit(channel_key: str, persona_key: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    try:
        res = (
            client.table("mvp_video_kits")
            .select("kit, created_at")
            .eq("channel_key", channel_key)
            .eq("persona_key", persona_key)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        items = getattr(res, "data", None) or []
        if not items:
            return None
        val = items[0].get("kit")
        return val if isinstance(val, dict) else None
    except Exception:
        return None


