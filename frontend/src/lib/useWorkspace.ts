import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from './auth';
import { useFiles } from './useFiles';
import { api, errorMessage } from './api';
import type { Conversation, Message, Turn } from './api';

export function useWorkspace() {
  const { client } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [active, setActive] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [pending, setPending] = useState('');
  const [mode, setMode] = useState<'hybrid' | 'vector' | 'text'>('hybrid');
  const [scope, setScope] = useState<'auto' | 'focused' | 'all_documents'>('auto');
  const initialized = useRef(false);
  const operation = useRef(false);
  const retry = useRef<{ fingerprint: string; conversationId: string; requestId: string } | null>(null);
  const activeRef = useRef<Conversation | null>(null);
  activeRef.current = active;
  const onReady = useCallback((id: string) => {
    if (!activeRef.current) setSelected(previous => [...new Set([...previous, id])].slice(0, 30));
  }, []);
  const files = useFiles(client, onReady);
  const documents = files.documents;
  const readySelected = selected.filter(id => documents.some(doc => doc.id === id && doc.status === 'ready') && !files.actions[id]);
  useEffect(() => {
    // Keep the user's selection through automatic model upgrades; only ready files inform a turn.
    setSelected(previous => previous.filter(id => documents.some(doc => doc.id === id && (doc.status === 'ready' || doc.status === 'processing'))));
  }, [documents]);

  const refresh = useCallback(async () => {
    if (!client) return;
    try {
      const [sources, history] = await Promise.all([
        files.refresh(), api<Conversation[]>(client, '/conversations'),
      ]);
      setConversations(history);
      if (!initialized.current) {
        setSelected(sources.filter(source => source.status === 'ready').slice(0, 30).map(source => source.id));
        initialized.current = true;
      }
    } catch (error) { setError(errorMessage(error)); } finally { setLoading(false); }
  }, [client, files.refresh]);
  useEffect(() => { void refresh(); }, [refresh]);
  const replaceConversation = (conversation: Conversation) => {
    setActive(conversation);
    setConversations(previous => [conversation, ...previous.filter(item => item.id !== conversation.id)]);
  };

  async function openConversation(id: string) {
    if (!client || operation.current) return;
    operation.current = true; setBusy(true); setError('');
    try {
      const result = await api<Turn>(client, `/conversations/${id}`);
      setActive(result.conversation); setMessages(result.messages); setSelected(result.conversation.document_ids.filter(id => documents.some(doc => doc.id === id && doc.status === 'ready')));
      retry.current = null;
    } catch (error) { setError(errorMessage(error)); } finally { operation.current = false; setBusy(false); }
  }
  function newConversation() {
    if (operation.current) return;
    setActive(null); setMessages([]); setError(''); retry.current = null;
    setSelected(previous => previous.filter(id => documents.some(source => source.id === id && source.status === 'ready')));
  }
  async function renameConversation(id: string, title: string) {
    if (!client || operation.current || !title.trim()) return;
    try {
      const updated = await api<Conversation>(client, `/conversations/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) });
      setConversations(previous => previous.map(item => item.id === id ? updated : item));
      if (active?.id === id) setActive(updated);
    } catch (error) { setError(errorMessage(error)); }
  }
  async function deleteConversation(id: string) {
    if (!client || operation.current) return;
    operation.current = true; setBusy(true);
    try {
      await api(client, `/conversations/${id}`, { method: 'DELETE' });
      setConversations(previous => previous.filter(item => item.id !== id));
      if (active?.id === id) { setActive(null); setMessages([]); retry.current = null; }
    } catch (error) { setError(errorMessage(error)); } finally { operation.current = false; setBusy(false); }
  }
  async function send(content: string) {
    if (!client || operation.current || !content.trim()) return false;
    const selectedForTurn = [...readySelected];
    if (!selectedForTurn.length) { setError('Select at least one ready source. Other files can keep processing in the background.'); return false; }
    operation.current = true; setBusy(true); setPending(content); setError('');
    try {
      let conversation = active;
      if (!conversation) {
        conversation = await api<Conversation>(client, '/conversations', { method: 'POST', body: JSON.stringify({ document_ids: selectedForTurn }) });
        replaceConversation(conversation);
      } else if (JSON.stringify(selectedForTurn) !== JSON.stringify(conversation.document_ids)) {
        conversation = await api<Conversation>(client, `/conversations/${conversation.id}`, { method: 'PATCH', body: JSON.stringify({ document_ids: selectedForTurn }) });
        replaceConversation(conversation);
      }
      const fingerprint = JSON.stringify({ content, selected: selectedForTurn, mode, scope });
      const requestId = retry.current?.fingerprint === fingerprint && retry.current.conversationId === conversation.id
        ? retry.current.requestId : crypto.randomUUID();
      retry.current = { fingerprint, conversationId: conversation.id, requestId };
      const result = await api<Turn>(client, `/conversations/${conversation.id}/chat`, { method: 'POST',
        body: JSON.stringify({ content, request_id: requestId, mode, scope }) });
      setMessages(previous => [...previous.filter(message => !result.messages.some(saved => saved.id === message.id)), ...result.messages]);
      replaceConversation(result.conversation); retry.current = null;
      void refresh();
      return true;
    } catch (error) {
      setError(errorMessage(error));
      // A model change may have queued reindexing while rejecting this question with 409.
      void files.refresh().catch(() => {});
      return false;
    }
    finally { operation.current = false; setBusy(false); setPending(''); }
  }
  function toggleSource(id: string) {
    if (operation.current) return;
    setSelected(previous => previous.includes(id) ? previous.filter(item => item !== id) : [...previous, id].slice(0, 30));
  }
  return { documents, conversations, active, messages, selected, readySelected, loading, busy, uploading: files.uploading,
    fileActions: files.actions, error, pending, mode, scope,
    setMode, setScope, setError, refresh, openConversation, newConversation, renameConversation, deleteConversation,
    upload: files.upload, retryDocument: files.retry, deleteDocument: files.remove, send, toggleSource };
}

