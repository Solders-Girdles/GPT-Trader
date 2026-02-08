-- Events table for durable event storage
-- Schema version: 1

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    bot_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_bot_id ON events(bot_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(json_extract(payload, '$.symbol'));

-- Schema version tracking for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);
