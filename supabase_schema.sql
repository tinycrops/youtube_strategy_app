-- YouTube Strategy Analyzer Database Schema
-- Copy and paste this into your Supabase SQL Editor and run it

-- Channel analytics table (stores fetched video data)
CREATE TABLE IF NOT EXISTS channel_analytics (
    channel_key TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Channel strategies table (stores AI-generated strategies)  
CREATE TABLE IF NOT EXISTS channel_strategies (
    channel_key TEXT PRIMARY KEY,
    strategy JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Creator personas table (stores creator personality profiles)
CREATE TABLE IF NOT EXISTS creator_personas (
    persona_key TEXT PRIMARY KEY,
    persona JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content journal table (stores content creation decisions)
CREATE TABLE IF NOT EXISTS content_journal (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    entry JSONB NOT NULL
);

-- Preference events table (stores user feedback for learning)
CREATE TABLE IF NOT EXISTS preference_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    event JSONB NOT NULL
);

-- MVP video kits table (stores generated video templates)
CREATE TABLE IF NOT EXISTS mvp_video_kits (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    channel_key TEXT NOT NULL,
    persona_key TEXT NOT NULL,
    kit JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add helpful indexes for better performance
CREATE INDEX IF NOT EXISTS idx_channel_analytics_updated_at ON channel_analytics(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_journal_created_at ON content_journal(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_preference_events_created_at ON preference_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mvp_video_kits_channel_key ON mvp_video_kits(channel_key);
CREATE INDEX IF NOT EXISTS idx_mvp_video_kits_persona_key ON mvp_video_kits(persona_key);