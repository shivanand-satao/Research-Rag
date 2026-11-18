import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import { ArrowUp, Library, LoaderCircle, Menu, RefreshCw, X } from 'lucide-react';
import { useAuth } from '../lib/auth';
import { useWorkspace } from '../lib/useWorkspace';
import { HistorySidebar } from '../components/HistorySidebar';
import { SourcePanel } from '../components/SourcePanel';
import { Message } from '../components/Message';

export function Workspace() {
  const auth = useAuth();
  const state = useWorkspace();
  const [draft, setDraft] = useState('');
  const [sidebar, setSidebar] = useState(false);
  const [sources, setSources] = useState(true);
  const bottom = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const disabled = state.busy || state.loading;
  const sourceCount = state.readySelected.length;
  const name = auth.session?.user.user_metadata.display_name || auth.session?.user.email?.split('@')[0] || 'Researcher';
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }); }, [state.messages, state.pending]);
  useEffect(() => { if (textarea.current) { textarea.current.style.height = 'auto'; textarea.current.style.height = `${Math.min(textarea.current.scrollHeight, 160)}px`; } }, [draft]);
  useEffect(() => { setDraft(''); }, [state.active?.id]);
  async function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!draft.trim() || disabled) return;
    const content = draft.trim();
    setDraft('');
    if (!await state.send(content)) setDraft(content);
  }
  function handleKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); }
  }
  async function logout() {
    const { error } = await auth.client!.auth.signOut();
    if (error) state.setError(error.message);
  }
  return <div className={`app-shell ${sources ? 'with-sources' : ''} ${sidebar ? 'sidebar-open' : ''}`}>
    {sidebar && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setSidebar(false)}/>}
    <HistorySidebar conversations={state.conversations} activeId={state.active?.id} disabled={disabled} name={name} email={auth.session?.user.email || ''}
      onNew={state.newConversation} onOpen={id => void state.openConversation(id)} onRename={(id, title) => void state.renameConversation(id, title)} onDelete={id => void state.deleteConversation(id)} onLogout={() => void logout()} onClose={() => setSidebar(false)}/>
    <main className="chat-workspace"><header className="chat-header"><div><button className="icon-button mobile-only" onClick={() => setSidebar(true)} aria-label="Open navigation"><Menu size={19}/></button><span className="header-breadcrumb">Workspace <span>/</span></span><h1>{state.active?.title || 'New research'}</h1></div><button className={`button sources-toggle ${sources ? 'active' : ''}`} onClick={() => setSources(!sources)}><Library size={16}/> Sources <span>{sourceCount}</span></button></header>
      {state.error && <div className="workspace-alert" role="alert"><span>{state.error}</span><button className="icon-button" onClick={() => { state.setError(''); void state.refresh(); }} aria-label="Retry loading workspace"><RefreshCw size={15}/></button><button className="icon-button" onClick={() => state.setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
      <div className="chat-scroll">{state.loading ? <div className="page-status"><LoaderCircle className="spin"/> Loading your research…</div> : state.messages.length === 0 && !state.pending ? <section className="chat-welcome"><h2>What would you like<br/>to research?</h2></section> : <div className="message-thread">{state.messages.map(message => <Message key={message.id} message={message} onError={state.setError}/>)}{state.pending && <><article className="message user-message"><span className="message-label">YOU</span><p>{state.pending}</p></article><div className="thinking" role="status"><span className="thinking-dots"><i/><i/><i/></span> Reading the evidence and connecting the ideas…</div></>}</div>}<div ref={bottom}/></div>
      <div className="composer-area"><form className="composer" onSubmit={event => void submit(event)}><textarea ref={textarea} aria-label="Ask a research question" value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={handleKey} placeholder="Ask a question. Follow a thought." rows={1} maxLength={6000} disabled={disabled}/><div className="composer-toolbar"><div><label className="select-control"><span className="status-dot"/><select aria-label="Retrieval mode" value={state.mode} disabled={disabled} onChange={event => state.setMode(event.target.value as typeof state.mode)}><option value="hybrid">Hybrid search</option><option value="vector">Semantic search</option><option value="text">Text search</option></select></label><label className="select-control scope-control"><select aria-label="Document retrieval scope" value={state.scope} disabled={disabled} onChange={event => state.setScope(event.target.value as typeof state.scope)}><option value="auto">Auto scope</option><option value="focused">Focused</option><option value="all_documents">Across all sources</option></select></label></div><button className="send-button" aria-label="Send question" disabled={!draft.trim() || disabled || !sourceCount}>{state.busy ? <LoaderCircle className="spin" size={18}/> : <ArrowUp size={20}/>}</button></div></form><div className="composer-note"><span>{sourceCount ? `${sourceCount} source${sourceCount === 1 ? '' : 's'} in focus` : 'Add a source to begin'}</span><span>Verify important claims in the original sources.</span><span>↵ Send <i>·</i> Shift ↵ New line</span></div></div>
    </main>
    {sources && <SourcePanel documents={state.documents} selected={state.readySelected} disabled={state.loading} selectionDisabled={disabled} uploading={state.uploading} fileActions={state.fileActions} onUpload={state.upload} onToggle={state.toggleSource} onDelete={state.deleteDocument} onRetry={state.retryDocument} onClose={() => setSources(false)}/>}
  </div>;
}
