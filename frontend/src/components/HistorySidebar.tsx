import { useState } from 'react';
import { ChevronDown, Edit3, LogOut, MessageSquare, Plus, Search, Trash2, X } from 'lucide-react';
import { Brand } from './Brand';
import type { Conversation } from '../lib/api';

interface Props { conversations: Conversation[]; activeId?: string; disabled: boolean; name: string; email: string; onNew: () => void; onOpen: (id: string) => void; onRename: (id: string, title: string) => void; onDelete: (id: string) => void; onLogout: () => void; onClose: () => void }

export function HistorySidebar({ conversations, activeId, disabled, name, email, onNew, onOpen, onRename, onDelete, onLogout, onClose }: Props) {
  const [search, setSearch] = useState('');
  const [editing, setEditing] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);
  const [account, setAccount] = useState(false);
  const filtered = conversations.filter(item => item.title.toLowerCase().includes(search.toLowerCase()));
  return <aside className="history-sidebar"><div className="sidebar-brand"><Brand/><button className="icon-button mobile-only" aria-label="Close navigation" onClick={onClose}><X size={19}/></button></div>
    <button className="button new-research" onClick={onNew} disabled={disabled}><Plus size={17}/> New research <span>↗</span></button>
    <div className="history-search"><Search size={15}/><input aria-label="Search chat history" placeholder="Find a conversation" value={search} onChange={event => setSearch(event.target.value)}/></div>
    <div className="sidebar-section-label">CONVERSATIONS <span>{conversations.length.toString().padStart(2, '0')}</span></div>
    <div className="conversation-list">{filtered.length === 0 ? <div className="history-empty"><MessageSquare size={22}/><p>{search ? 'No conversations match.' : 'A fresh page for your ideas.'}</p><span>{search ? 'Try a different search.' : 'Your saved research will appear here.'}</span></div> : filtered.map(item => <div key={item.id} className={`history-item ${activeId === item.id ? 'active' : ''}`}>
      {editing === item.id ? <form onSubmit={event => { event.preventDefault(); onRename(item.id, title); setEditing(null); }}><input aria-label="Conversation title" autoFocus value={title} maxLength={120} onChange={event => setTitle(event.target.value)} onKeyDown={event => { if (event.key === 'Escape') setEditing(null); }}/><button disabled={disabled || !title.trim()}>Save</button><button type="button" onClick={() => setEditing(null)}>Cancel</button></form> : <><button className="history-open" disabled={disabled} onClick={() => { onOpen(item.id); onClose(); }}><MessageSquare size={15}/><span>{item.title}</span></button><div className="history-item-actions"><button className="icon-button" disabled={disabled} aria-label={`Rename ${item.title}`} onClick={() => { setEditing(item.id); setTitle(item.title); }}><Edit3 size={12}/></button><button className="icon-button" disabled={disabled} aria-label={`Delete ${item.title}`} onClick={() => setDeleting(item.id)}><Trash2 size={12}/></button></div></>}
      {deleting === item.id && <div className="inline-confirm"><span>Delete this chat?</span><button disabled={disabled} onClick={() => { onDelete(item.id); setDeleting(null); }}>Delete</button><button onClick={() => setDeleting(null)}>Cancel</button></div>}
    </div>)}</div>
    <div className="sidebar-bottom"><div className="workspace-label"><span className="status-dot"/> PERSONAL WORKSPACE</div>{account && <div className="account-menu"><small>{email}</small><button disabled={disabled} onClick={onLogout}><LogOut size={15}/> Log out</button></div>}<button className="account-button" onClick={() => setAccount(!account)}><span className="avatar">{name.slice(0, 1).toUpperCase()}</span><span><strong>{name}</strong><small>Researcher</small></span><ChevronDown size={15}/></button></div>
  </aside>;
}

