-- Research RAG owns a dedicated set of tables and one private storage bucket.
begin;

create table public.research_documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null check (length(btrim(title)) between 1 and 180),
  source text not null,
  mime_type text not null,
  storage_path text not null,
  size_bytes bigint not null check (size_bytes > 0 and size_bytes <= 26214400),
  status text not null default 'processing' check (status in ('processing', 'ready', 'failed', 'deleting')),
  chunk_count integer not null default 0 check (chunk_count >= 0),
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  check (storage_path like user_id::text || '/' || id::text || '/%')
);

create table public.research_conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default 'New research' check (length(btrim(title)) between 1 and 120),
  document_ids uuid[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  check (cardinality(document_ids) <= 30)
);

create table public.research_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  request_id uuid not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (length(btrim(content)) > 0),
  metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
  created_at timestamptz not null default clock_timestamp(),
  foreign key (conversation_id, user_id) references public.research_conversations(id, user_id) on delete cascade,
  unique (conversation_id, request_id, role)
);

create index research_documents_owner_idx on public.research_documents(user_id, created_at desc);
create index research_conversations_owner_idx on public.research_conversations(user_id, updated_at desc);
create index research_messages_history_idx on public.research_messages(conversation_id, created_at, id);
create index research_messages_owner_idx on public.research_messages(user_id);

create function public.research_touch_updated_at() returns trigger
language plpgsql set search_path = '' as $$
begin new.updated_at = clock_timestamp(); return new; end;
$$;
create trigger research_documents_touch before update on public.research_documents
for each row execute function public.research_touch_updated_at();
create trigger research_conversations_touch before update on public.research_conversations
for each row execute function public.research_touch_updated_at();

create function public.research_check_document_scope() returns trigger
language plpgsql set search_path = '' as $$
begin
  -- Removing sources is always allowed, including sources currently being deleted.
  if tg_op = 'UPDATE' and new.document_ids <@ old.document_ids then return new; end if;
  if exists (select 1 from unnest(new.document_ids) as selected(id)
      where not exists (select 1 from public.research_documents d
        where d.id = selected.id and d.user_id = new.user_id and d.status = 'ready')) then
    raise exception 'Source selection contains unavailable documents' using errcode = '23514';
  end if;
  return new;
end;
$$;
create trigger research_conversations_check_sources before insert or update of document_ids
on public.research_conversations for each row execute function public.research_check_document_scope();

-- Remove deleted sources from selections; historical answer excerpts remain saved.
create function public.research_remove_document_scope() returns trigger
language plpgsql set search_path = '' as $$
begin
  update public.research_conversations set document_ids = array_remove(document_ids, old.id)
  where user_id = old.user_id and old.id = any(document_ids);
  return old;
end;
$$;
create trigger research_documents_remove_scope after delete on public.research_documents
for each row execute function public.research_remove_document_scope();

alter table public.research_documents enable row level security;
alter table public.research_conversations enable row level security;
alter table public.research_messages enable row level security;

create policy research_documents_own on public.research_documents for all to authenticated
using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy research_conversations_own on public.research_conversations for all to authenticated
using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy research_messages_own on public.research_messages for all to authenticated
using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));

grant select, insert, update, delete on public.research_documents, public.research_conversations, public.research_messages to authenticated;
revoke all on public.research_documents, public.research_conversations, public.research_messages from anon;

-- Atomic, idempotent question + answer save. No service-role credentials required.
create function public.research_save_turn(p_conversation_id uuid, p_request_id uuid,
  p_question text, p_answer text, p_metadata jsonb) returns jsonb
language plpgsql security invoker set search_path = '' as $$
declare
  c public.research_conversations;
  turn_time timestamptz := clock_timestamp();
begin
  select * into c from public.research_conversations
    where id = p_conversation_id and user_id = (select auth.uid()) for update;
  if not found then raise exception 'Conversation not found' using errcode = '42501'; end if;
  if not exists (select 1 from public.research_messages
      where conversation_id = p_conversation_id and request_id = p_request_id) then
    insert into public.research_messages(conversation_id, user_id, request_id, role, content, metadata, created_at)
    values (c.id, c.user_id, p_request_id, 'user', p_question, '{}'::jsonb, turn_time),
           (c.id, c.user_id, p_request_id, 'assistant', p_answer, p_metadata, turn_time + interval '1 microsecond');
    update public.research_conversations set
      title = case when title = 'New research' then left(p_question, 80) else title end
      where id = c.id returning * into c;
  end if;
  return jsonb_build_object('conversation', to_jsonb(c), 'messages', (
    select jsonb_agg(to_jsonb(m) order by m.created_at, m.id) from public.research_messages m
    where m.conversation_id = c.id and m.request_id = p_request_id));
end;
$$;
revoke execute on function public.research_save_turn(uuid, uuid, text, text, jsonb) from public, anon;
grant execute on function public.research_save_turn(uuid, uuid, text, text, jsonb) to authenticated;
revoke execute on function public.research_touch_updated_at(), public.research_check_document_scope(), public.research_remove_document_scope() from public, anon, authenticated;

insert into storage.buckets(id, name, public, file_size_limit, allowed_mime_types)
values ('research-documents', 'research-documents', false, 26214400,
  array['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown']);

create policy research_storage_read on storage.objects for select to authenticated
using (bucket_id = 'research-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy research_storage_insert on storage.objects for insert to authenticated
with check (bucket_id = 'research-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy research_storage_delete on storage.objects for delete to authenticated
using (bucket_id = 'research-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);

commit;
