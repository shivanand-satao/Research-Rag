import { useCallback, useEffect, useRef, useState } from 'react';
import type { SupabaseClient } from '@supabase/supabase-js';
import { api, errorMessage } from './api';
import type { DocumentSource, FileAction } from './api';

interface UploadTask { file: File; id: string; done: () => void }

export function useFiles(client: SupabaseClient | null, onReady: (id: string) => void) {
  const [documents, setDocuments] = useState<DocumentSource[]>([]);
  const [actions, setActions] = useState<Record<string, FileAction>>({});
  const [uploading, setUploading] = useState(false);
  const sources = useRef<DocumentSource[]>([]);
  const localFiles = useRef(new Map<string, File>());
  const inFlight = useRef(new Set<string>());
  const queue = useRef<UploadTask[]>([]);
  const activeUploads = useRef(0);
  const refreshPromise = useRef<Promise<DocumentSource[]> | null>(null);
  const localErrors = useRef(new Map<string, string>());

  const commit = useCallback((next: DocumentSource[]) => {
    sources.current = next;
    setDocuments(next);
  }, []);
  const update = (id: string, patch: Partial<DocumentSource>) => commit(sources.current.map(doc => doc.id === id ? { ...doc, ...patch } : doc));
  const action = (id: string, value?: FileAction) => setActions(previous => {
    const next = { ...previous };
    if (value) next[id] = value; else delete next[id];
    return next;
  });

  const refresh = useCallback(async () => {
    if (!client) return sources.current;
    if (refreshPromise.current) return refreshPromise.current;
    refreshPromise.current = (async () => {
      const incoming = await api<DocumentSource[]>(client, '/documents');
      const becameReady = incoming.filter(doc => doc.status === 'ready' && sources.current.some(old => old.id === doc.id && old.status !== 'ready'));
      becameReady.forEach(doc => localErrors.current.delete(doc.id));
      const local = sources.current.filter(doc => doc.id.startsWith('local:'));
      commit([...local, ...incoming.map(doc => ({ ...doc, error: localErrors.current.get(doc.id) ?? doc.error }))]);
      becameReady.forEach(doc => onReady(doc.id));
      return incoming;
    })().finally(() => { refreshPromise.current = null; });
    return refreshPromise.current;
  }, [client, commit, onReady]);

  useEffect(() => {
    if (!documents.some(doc => doc.status === 'processing' || (doc.status === 'deleting' && !doc.error))) return;
    const timer = window.setInterval(() => { void refresh().catch(() => {}); }, 1500);
    return () => window.clearInterval(timer);
  }, [documents, refresh]);

  function pump() {
    if (!client) return;
    while (activeUploads.current < 2 && queue.current.length) {
      const task = queue.current.shift()!;
      activeUploads.current++;
      setUploading(true);
      action(task.id, 'uploading');
      update(task.id, { activity: { phase: 'Uploading original', progress: 0 } });
      const form = new FormData(); form.append('file', task.file);
      void api<DocumentSource>(client, '/documents', { method: 'POST', body: form }).then(source => {
        commit([source, ...sources.current.filter(doc => doc.id !== task.id && doc.id !== source.id)]);
        localFiles.current.delete(task.id);
      }).catch(error => {
        update(task.id, { status: 'failed', error: errorMessage(error), activity: undefined });
      }).finally(() => {
        action(task.id); activeUploads.current--; task.done();
        setUploading(activeUploads.current > 0 || queue.current.length > 0);
        pump();
        void refresh().catch(() => {});
      });
    }
  }

  async function upload(files: FileList | File[]) {
    if (!client) return;
    const pending = Array.from(files).map(file => new Promise<void>(done => {
      const id = `local:${crypto.randomUUID()}`;
      const invalid = !/\.(pdf|docx|txt|md)$/i.test(file.name) ? 'Choose a PDF, DOCX, TXT, or Markdown file.'
        : !file.size || file.size > 25 * 1024 * 1024 ? 'Choose a nonempty file smaller than 25 MB.' : null;
      localFiles.current.set(id, file);
      commit([{ id, title: file.name, source: file.name, status: invalid ? 'failed' : 'processing',
        chunk_count: 0, size_bytes: file.size, created_at: new Date().toISOString(), error: invalid,
        ...(!invalid ? { activity: { phase: 'Waiting to upload', progress: 0 } } : {}) }, ...sources.current]);
      if (invalid) done(); else queue.current.push({ id, file, done });
    }));
    pump();
    await Promise.all(pending);
  }

  async function retry(id: string, replacement?: File) {
    if (!client || inFlight.current.has(id)) return;
    if (id.startsWith('local:')) {
      const file = replacement ?? localFiles.current.get(id);
      if (file) {
        commit(sources.current.filter(doc => doc.id !== id)); localFiles.current.delete(id);
        await upload([file]);
      }
      return;
    }
    inFlight.current.add(id); action(id, 'retrying');
    localErrors.current.delete(id);
    update(id, { error: null, activity: { phase: 'Queuing retry', progress: 0 } });
    try {
      const form = replacement ? new FormData() : undefined;
      if (form && replacement) form.append('file', replacement);
      const source = await api<DocumentSource>(client, `/documents/${id}/retry`, { method: 'POST', body: form });
      update(id, source);
    } catch (error) { localErrors.current.set(id, errorMessage(error)); update(id, { error: errorMessage(error), activity: undefined }); }
    finally { inFlight.current.delete(id); action(id); }
  }

  async function remove(id: string) {
    if (!client || inFlight.current.has(id)) return;
    if (id.startsWith('local:')) {
      if (actions[id] === 'uploading') return;
      const pending = queue.current.find(task => task.id === id);
      queue.current = queue.current.filter(task => task.id !== id); pending?.done();
      localFiles.current.delete(id); commit(sources.current.filter(doc => doc.id !== id));
      return;
    }
    inFlight.current.add(id); action(id, 'deleting');
    localErrors.current.delete(id);
    update(id, { error: null });
    try {
      const source = await api<DocumentSource>(client, `/documents/${id}`, { method: 'DELETE' });
      update(id, source);
      void refresh().catch(() => {});
    } catch (error) { localErrors.current.set(id, errorMessage(error)); update(id, { error: errorMessage(error) }); }
    finally { inFlight.current.delete(id); action(id); }
  }

  return { documents, actions, uploading, refresh, upload, retry, remove };
}
