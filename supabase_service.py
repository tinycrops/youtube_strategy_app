import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

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
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime().isoformat()
            elif hasattr(v, "isoformat") and callable(getattr(v, "isoformat", None)):
                try:
                    v = v.isoformat()
                except Exception:
                    pass
            if isinstance(v, (pd.Int64Dtype, )):
                v = int(v)
            fixed[k] = v
        serializable.append(fixed)
    return serializable


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
        "updated_at": datetime.now(timezone.utc).isoformat(),
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


def upsert_strategy(channel_key: str, strategy: Dict[str, Any]) -> None:
    client = get_client()
    payload = {
        "channel_key": channel_key,
        "strategy": strategy or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
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


