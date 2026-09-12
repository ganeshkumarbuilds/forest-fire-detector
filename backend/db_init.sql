-- Forest Fire Detector — Postgres manual setup
-- Run once in psql / pgAdmin / Neon SQL editor.
-- You can assign values manually with the examples at the bottom.

CREATE TABLE IF NOT EXISTS detections (
  id SERIAL PRIMARY KEY,
  filename TEXT NOT NULL DEFAULT '',
  fire_detected BOOLEAN NOT NULL DEFAULT FALSE,
  confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alert_config (
  id INT PRIMARY KEY CHECK (id = 1),
  email TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS hotspots_cache (
  id INT PRIMARY KEY CHECK (id = 1),
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ===== Assign YOUR OWN values manually =====

-- 1) Alert email (only id=1 is used by the app):
-- INSERT INTO alert_config (id, email) VALUES (1, 'you@example.com')
-- ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email;

-- 2) Manual detection entry:
-- INSERT INTO detections (filename, fire_detected, confidence, timestamp)
-- VALUES ('Camera 11 — manual test.jpg', true, 0.97, NOW());

-- 3) Check data:
-- SELECT * FROM detections ORDER BY id DESC LIMIT 10;
-- SELECT * FROM alert_config;
-- SELECT updated_at, payload->>'count' AS count FROM hotspots_cache WHERE id = 1;

-- 4) Clear test data (optional):
-- DELETE FROM detections;
