-- GPT-Trader PostgreSQL Schema Initialization
-- Phase 2.5 - Production Database Setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Note: TimescaleDB extension is automatically enabled in the Docker image

-- Create schemas for logical separation
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS portfolio;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO trading, ml, portfolio, monitoring, public;

-- Create custom types
CREATE TYPE trading.order_side AS ENUM ('buy', 'sell');
CREATE TYPE trading.order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');
CREATE TYPE trading.order_status AS ENUM ('pending', 'submitted', 'partial', 'filled', 'cancelled', 'rejected', 'expired');
CREATE TYPE trading.position_status AS ENUM ('open', 'closed', 'partial');

-- Grant permissions (for production, use more restrictive permissions)
GRANT USAGE ON SCHEMA trading TO trader;
GRANT USAGE ON SCHEMA ml TO trader;
GRANT USAGE ON SCHEMA portfolio TO trader;
GRANT USAGE ON SCHEMA monitoring TO trader;

GRANT CREATE ON SCHEMA trading TO trader;
GRANT CREATE ON SCHEMA ml TO trader;
GRANT CREATE ON SCHEMA portfolio TO trader;
GRANT CREATE ON SCHEMA monitoring TO trader;
