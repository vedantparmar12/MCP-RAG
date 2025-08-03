-- Enable the pgvector extension
create extension if not exists vector;

-- Drop tables if they exist (to allow rerunning the script)
drop table if exists crawled_pages;
drop table if exists code_examples;
drop table if exists sources;
drop table if exists visual_documents;

-- Create the sources table
create table sources (
    source_id text primary key,
    summary text,
    total_word_count integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create the documentation chunks table with flexible embedding dimensions
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding_provider text not null default 'openai',  -- Track which provider was used
    embedding_model text not null default 'text-embedding-3-small',  -- Track which model was used
    embedding_dimension integer not null default 1536,  -- Track embedding dimension
    embedding vector,  -- No fixed dimension - will be determined at insert time
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    foreign key (source_id) references sources(source_id)
);

-- Create indexes for better performance
create index idx_crawled_pages_embedding_openai on crawled_pages using ivfflat ((embedding::vector(1536)) vector_cosine_ops) where embedding_dimension = 1536;
create index idx_crawled_pages_embedding_cohere on crawled_pages using ivfflat ((embedding::vector(1024)) vector_cosine_ops) where embedding_dimension = 1024;
create index idx_crawled_pages_embedding_ollama on crawled_pages using ivfflat ((embedding::vector(768)) vector_cosine_ops) where embedding_dimension = 768;
create index idx_crawled_pages_embedding_minilm on crawled_pages using ivfflat ((embedding::vector(384)) vector_cosine_ops) where embedding_dimension = 384;

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages (source_id);

-- Create indexes on embedding provider and model for filtering
CREATE INDEX idx_crawled_pages_provider ON crawled_pages (embedding_provider);
CREATE INDEX idx_crawled_pages_model ON crawled_pages (embedding_model);

-- Create a flexible function to search for documentation chunks
create or replace function match_crawled_pages_flexible (
  query_embedding vector,
  embedding_dim int,
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL,
  provider_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  embedding_provider text,
  embedding_model text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    source_id,
    embedding_provider,
    embedding_model,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
    AND embedding_dimension = embedding_dim
    AND (source_filter IS NULL OR source_id = source_filter)
    AND (provider_filter IS NULL OR embedding_provider = provider_filter)
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Keep the original function for backward compatibility
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select id, url, chunk_number, content, metadata, source_id, similarity
  from match_crawled_pages_flexible(
    query_embedding, 
    1536, 
    match_count, 
    filter, 
    source_filter,
    'openai'
  );
end;
$$;

-- Enable RLS on the crawled_pages table
alter table crawled_pages enable row level security;

-- Create a policy that allows anyone to read crawled_pages
create policy "Allow public read access to crawled_pages"
  on crawled_pages
  for select
  to public
  using (true);

-- Enable RLS on the sources table
alter table sources enable row level security;

-- Create a policy that allows anyone to read sources
create policy "Allow public read access to sources"
  on sources
  for select
  to public
  using (true);

-- Create the code_examples table with flexible embedding dimensions
create table code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- The code example content
    summary text not null,  -- Summary of the code example
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding_provider text not null default 'openai',
    embedding_model text not null default 'text-embedding-3-small',
    embedding_dimension integer not null default 1536,
    embedding vector,  -- No fixed dimension
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    foreign key (source_id) references sources(source_id)
);

-- Create indexes for better vector similarity search performance
create index idx_code_examples_embedding_openai on code_examples using ivfflat ((embedding::vector(1536)) vector_cosine_ops) where embedding_dimension = 1536;
create index idx_code_examples_embedding_cohere on code_examples using ivfflat ((embedding::vector(1024)) vector_cosine_ops) where embedding_dimension = 1024;
create index idx_code_examples_embedding_ollama on code_examples using ivfflat ((embedding::vector(768)) vector_cosine_ops) where embedding_dimension = 768;
create index idx_code_examples_embedding_minilm on code_examples using ivfflat ((embedding::vector(384)) vector_cosine_ops) where embedding_dimension = 384;

-- Create an index on metadata for faster filtering
create index idx_code_examples_metadata on code_examples using gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_code_examples_source_id ON code_examples (source_id);

-- Create a flexible function to search for code examples
create or replace function match_code_examples_flexible (
  query_embedding vector,
  embedding_dim int,
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL,
  provider_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  embedding_provider text,
  embedding_model text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    summary,
    metadata,
    source_id,
    embedding_provider,
    embedding_model,
    1 - (code_examples.embedding <=> query_embedding) as similarity
  from code_examples
  where metadata @> filter
    AND embedding_dimension = embedding_dim
    AND (source_filter IS NULL OR source_id = source_filter)
    AND (provider_filter IS NULL OR embedding_provider = provider_filter)
  order by code_examples.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Keep the original function for backward compatibility
create or replace function match_code_examples (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select id, url, chunk_number, content, summary, metadata, source_id, similarity
  from match_code_examples_flexible(
    query_embedding,
    1536,
    match_count,
    filter,
    source_filter,
    'openai'
  );
end;
$$;

-- Enable RLS on the code_examples table
alter table code_examples enable row level security;

-- Create a policy that allows anyone to read code_examples
create policy "Allow public read access to code_examples"
  on code_examples
  for select
  to public
  using (true);

-- Create the visual_documents table for ColPali embeddings
create table visual_documents (
    id bigserial primary key,
    doc_id text unique not null,
    image_path text not null,
    text_content text,  -- Optional OCR or provided text
    metadata jsonb not null default '{}'::jsonb,
    source_id text,
    -- ColPali uses patch embeddings, stored as array
    patch_embeddings jsonb not null,  -- Array of embeddings for each patch
    num_patches integer not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add foreign key constraint to sources table if source_id is provided
    foreign key (source_id) references sources(source_id)
);

-- Create indexes for visual documents
create index idx_visual_documents_doc_id on visual_documents (doc_id);
create index idx_visual_documents_metadata on visual_documents using gin (metadata);
create index idx_visual_documents_source_id on visual_documents (source_id);

-- Enable RLS on the visual_documents table
alter table visual_documents enable row level security;

-- Create a policy that allows anyone to read visual_documents
create policy "Allow public read access to visual_documents"
  on visual_documents
  for select
  to public
  using (true);

-- Create a view to show embedding statistics
CREATE OR REPLACE VIEW embedding_statistics AS
SELECT 
    'crawled_pages' as table_name,
    embedding_provider,
    embedding_model,
    embedding_dimension,
    COUNT(*) as count
FROM crawled_pages
GROUP BY embedding_provider, embedding_model, embedding_dimension

UNION ALL

SELECT 
    'code_examples' as table_name,
    embedding_provider,
    embedding_model,
    embedding_dimension,
    COUNT(*) as count
FROM code_examples
GROUP BY embedding_provider, embedding_model, embedding_dimension

ORDER BY table_name, embedding_provider, embedding_model;