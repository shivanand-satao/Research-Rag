import type { SupabaseClient } from '@supabase/supabase-js';
import { apiUrl, responseError } from './endpoints';

export interface DocumentSource { id: string; title: string; source: string; status: 'ready' | 'processing' | 'failed' | 'deleting'; chunk_count: number; size_bytes: number; created_at: string; error: string | null; activity?: { phase: string; progress: number } }
export type FileAction = 'uploading' | 'retrying' | 'deleting';
export interface Conversation { id: string; title: string; document_ids: string[]; updated_at: string; created_at: string }
export interface Citation { number: number; document_id: string; source: string; page: number | null; section: string | null; excerpt: string }
export interface Message { id: string; role: 'user' | 'assistant'; content: string; created_at: string; request_id?: string; metadata: { citations?: Citation[]; grounded?: boolean; model?: string } }
export interface Turn { conversation: Conversation; messages: Message[] }

export async function api<T>(client: SupabaseClient, path: string, init: RequestInit = {}): Promise<T> {
  const { data: { session } } = await client.auth.getSession();
  if (!session) throw new Error('Your session expired. Please sign in again.');
  const response = await fetch(apiUrl(path), {
    ...init, headers: { Authorization: `Bearer ${session.access_token}`,
      ...(init.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...init.headers },
  });
  if (!response.ok) {
    throw new Error(await responseError(response));
  }
  return response.status === 204 ? undefined as T : response.json();
}

export const documentViewerUrl = (id: string, page?: number | null) => `/documents/${encodeURIComponent(id)}${page ? `?page=${page}` : ''}`;

export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please retry.';

