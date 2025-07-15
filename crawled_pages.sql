-- Run this in your Supabase SQL Editor to fix the dimension mismatch

-- First, drop existing tables to avoid conflicts
DROP TABLE IF EXISTS crawled_pages CASCADE;
DROP TABLE IF EXISTS code_examples CASCADE;
DROP TABLE IF EXISTS sources CASCADE;

-- Recreate sources table
CREATE TABLE sources (
    source_id text primary key,
    summary text,
    total_word_count integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Recreate crawled_pages table with 768 dimensions for Gemini
CREATE TABLE crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(768),  -- Changed from 1536 to 768 for Gemini
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id)
);

-- Recreate code_examples table with 768 dimensions
CREATE TABLE code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    summary text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(768),  -- Changed from 1536 to 768 for Gemini
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id)
);

-- Create indexes
CREATE INDEX ON crawled_pages USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_crawled_pages_metadata ON crawled_pages USING gin (metadata);
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages (source_id);

CREATE INDEX ON code_examples USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_code_examples_metadata ON code_examples USING gin (metadata);
CREATE INDEX idx_code_examples_source_id ON code_examples (source_id);

-- Update search functions for 768 dimensions
CREATE OR REPLACE FUNCTION match_crawled_pages (
  query_embedding vector(768),  -- Changed from 1536 to 768
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN query
  SELECT
    id,
    url,
    chunk_number,
    content,
    metadata,
    source_id,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  FROM crawled_pages
  WHERE metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  ORDER BY crawled_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION match_code_examples (
  query_embedding vector(768),  -- Changed from 1536 to 768
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN query
  SELECT
    id,
    url,
    chunk_number,
    content,
    summary,
    metadata,
    source_id,
    1 - (code_examples.embedding <=> query_embedding) as similarity
  FROM code_examples
  WHERE metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  ORDER BY code_examples.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Enable RLS and create policies
ALTER TABLE crawled_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_examples ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to crawled_pages"
  ON crawled_pages FOR SELECT TO public USING (true);

CREATE POLICY "Allow public read access to sources"
  ON sources FOR SELECT TO public USING (true);

CREATE POLICY "Allow public read access to code_examples"
  ON code_examples FOR SELECT TO public USING (true);