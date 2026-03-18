create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists public.documents_metadata (
    id uuid primary key default gen_random_uuid(),
    source_file_name text not null,
    document_type text,
    version_tag text not null default 'current',
    category text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (source_file_name, version_tag)
);

alter table public.documents_metadata
    drop column if exists source_path,
    drop column if exists extra,
    drop column if exists chunk_count;

create table if not exists public.documents_embeddings (
    id uuid primary key default gen_random_uuid(),
    metadata_id uuid not null references public.documents_metadata(id) on delete cascade,
    content text not null,
    chunk_index integer not null,
    embedding vector(1024) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.documents_embeddings
    drop column if exists chunk_tag;

alter table public.documents_embeddings
    drop constraint if exists documents_embeddings_chunk_index_check,
    add constraint documents_embeddings_chunk_index_check check (chunk_index >= 0);

alter table public.documents_embeddings
    drop constraint if exists documents_embeddings_metadata_id_chunk_index_key,
    add constraint documents_embeddings_metadata_id_chunk_index_key unique (metadata_id, chunk_index);

drop index if exists public.idx_documents_embeddings_hnsw;

create index if not exists idx_documents_embeddings_hnsw
on public.documents_embeddings
using hnsw (embedding vector_cosine_ops)
with (m = 32, ef_construction = 128);

create index if not exists idx_documents_embeddings_metadata_chunk
on public.documents_embeddings (metadata_id, chunk_index);

create index if not exists idx_documents_metadata_document_type
on public.documents_metadata (document_type);

create index if not exists idx_documents_metadata_category
on public.documents_metadata (category);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_documents_metadata_updated_at on public.documents_metadata;
create trigger trg_documents_metadata_updated_at
before update on public.documents_metadata
for each row execute function public.set_updated_at();

drop trigger if exists trg_documents_embeddings_updated_at on public.documents_embeddings;
create trigger trg_documents_embeddings_updated_at
before update on public.documents_embeddings
for each row execute function public.set_updated_at();

create or replace function public.match_documents(
    query_embedding vector(1024),
    match_count integer default 5,
    min_similarity float default 0.3
)
returns table (
    id uuid,
    metadata_id uuid,
    content text,
    source_file_name text,
    document_type text,
    version_tag text,
    category text,
    chunk_index integer,
    distance float,
    cosine_similarity float,
    similarity_0_1 float
)
language sql
stable
set hnsw.ef_search = 100
as $$
    with ranked as (
        select
            de.id,
            de.metadata_id,
            de.content,
            dm.source_file_name,
            dm.document_type,
            dm.version_tag,
            dm.category,
            de.chunk_index,
            (de.embedding <=> query_embedding) as distance
        from public.documents_embeddings de
        join public.documents_metadata dm
            on dm.id = de.metadata_id
    ),
    scored as (
        select
            id,
            metadata_id,
            content,
            source_file_name,
            document_type,
            version_tag,
            category,
            chunk_index,
            distance,
            1 - distance as cosine_similarity,
            1 - (distance / 2) as similarity_0_1
        from ranked
    )
    select
        id,
        metadata_id,
        content,
        source_file_name,
        document_type,
        version_tag,
        category,
        chunk_index,
        distance,
        cosine_similarity,
        similarity_0_1
    from scored
    where similarity_0_1 >= least(greatest(min_similarity, 0.0), 1.0)
    order by distance
    limit match_count;
$$;

-- RLS is intentionally disabled for now. It can be reintroduced later.
alter table public.documents_metadata disable row level security;
alter table public.documents_embeddings disable row level security;

drop policy if exists "service role full access metadata" on public.documents_metadata;
drop policy if exists "service role full access embeddings" on public.documents_embeddings;
drop policy if exists "deny all metadata anon" on public.documents_metadata;
drop policy if exists "deny all metadata authenticated" on public.documents_metadata;
drop policy if exists "deny all embeddings anon" on public.documents_embeddings;
drop policy if exists "deny all embeddings authenticated" on public.documents_embeddings;

