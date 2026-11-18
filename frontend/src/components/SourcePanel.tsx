import { useRef, useState } from 'react';
import { Check, ExternalLink, FileText, LoaderCircle, Plus, RotateCcw, Trash2, Upload, X } from 'lucide-react';
import { documentViewerUrl } from '../lib/api';
import type { DocumentSource, FileAction } from '../lib/api';

interface Props { documents: DocumentSource[]; selected: string[]; disabled: boolean; selectionDisabled: boolean; uploading: boolean; fileActions: Record<string, FileAction>; onUpload: (files: FileList | File[]) => Promise<void>; onToggle: (id: string) => void; onDelete: (id: string) => Promise<void>; onRetry: (id: string, replacement?: File) => Promise<void>; onClose: () => void }

export function SourcePanel({ documents, selected, disabled, selectionDisabled, uploading, fileActions = {}, onUpload, onToggle, onDelete, onRetry, onClose }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const replacementInput = useRef<HTMLInputElement>(null);
  const [replacementId, setReplacementId] = useState<string | null>(null);
  const processingCount = documents.filter(doc => doc.status === 'processing').length;
  return <aside className="source-panel" aria-label="Source library"><div className="panel-heading"><div><span className="eyebrow">YOUR EVIDENCE</span><h2>Sources <span>{documents.length}</span></h2></div><button className="icon-button" onClick={onClose} aria-label="Close source panel"><X size={19}/></button></div>
    <p className="panel-description">Select the documents you want to explore. Only selected sources inform your answers.</p>
    <input ref={input} type="file" multiple accept=".pdf,.docx,.txt,.md" hidden onChange={event => { if (event.target.files) void onUpload(event.target.files); event.target.value = ''; }}/>
    <input ref={replacementInput} type="file" accept=".pdf,.docx,.txt,.md" hidden onChange={event => {
      const file = event.target.files?.[0];
      if (file && replacementId) void onRetry(replacementId, file);
      event.target.value = ''; setReplacementId(null);
    }}/>
    <button className={`upload-zone ${dragging ? 'dragging' : ''}`} disabled={disabled} onClick={() => input.current?.click()}
      onDragOver={event => { event.preventDefault(); if (!disabled) setDragging(true); }} onDragLeave={() => setDragging(false)}
      onDrop={event => { event.preventDefault(); setDragging(false); if (!disabled) void onUpload(event.dataTransfer.files); }}>
      {uploading ? <LoaderCircle size={21} className="spin"/> : <Upload size={21}/>}<strong>{uploading ? 'Uploading files…' : 'Add your documents'}</strong><span>Drop files here or browse to add more</span><small>PDF, DOCX, TXT, MD · Up to 25 MB each</small>
    </button>
    {processingCount > 0 && <p className="file-background-note" role="status">{processingCount} file{processingCount === 1 ? '' : 's'} processing in the background. You can keep working.</p>}
    <div className="source-list-header"><span>{selected.length} SELECTED</span><span>PRIVATE LIBRARY</span></div>
    <div className="source-list">{documents.length === 0 ? <div className="source-empty"><FileText size={29}/><p>Start with a source.</p><span>Your research library will grow here.</span></div> : documents.map(source => {
      const ownAction = fileActions[source.id];
      const deletingNow = source.status === 'deleting' && !!source.activity && !source.error;
      const actionDisabled = disabled || !!ownAction || deletingNow;
      const local = source.id.startsWith('local:');
      const working = (source.status === 'processing' || source.status === 'deleting') && !source.error;
      return <article key={source.id} className={`source-item ${selected.includes(source.id) ? 'selected' : ''}`}>
      <div className="source-select"><button className="checkbox source-checkbox" disabled={selectionDisabled || !!ownAction || source.status !== 'ready'} onClick={() => onToggle(source.id)} aria-label={`${selected.includes(source.id) ? 'Deselect' : 'Select'} ${source.source}`} aria-pressed={selected.includes(source.id)}>{selected.includes(source.id) && <Check size={12}/>}</button><FileText size={17}/><span className="source-info">{local ? <strong>{source.title}</strong> : <a href={documentViewerUrl(source.id)} target="_blank" rel="noopener noreferrer" className="source-title">{source.title}</a>}<small>{source.status === 'ready' ? `${source.chunk_count} passages · ${(source.size_bytes / 1024).toFixed(0)} KB` : working ? source.activity?.phase || 'Queued' : source.status === 'deleting' ? 'Delete failed · Retry available' : 'Indexing failed · Retry available'}</small></span></div>
      {working && <div className="file-progress" role="progressbar" aria-label={`${source.title} processing progress`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={source.activity?.progress ?? 0}><span style={{ width: `${source.activity?.progress ?? 0}%` }}/></div>}
      {source.error && <p className="source-error">{source.error}</p>}
      <div className="source-actions">{!local && <a className="icon-button" aria-label={`Open ${source.source}`} href={documentViewerUrl(source.id)} target="_blank" rel="noopener noreferrer"><ExternalLink size={13}/></a>}
        {source.status === 'failed' && <><button className="icon-button" disabled={actionDisabled} aria-label={`Retry ${local ? 'uploading' : 'indexing'} ${source.source}`} title={local ? 'Retry upload' : 'Retry indexing'} onClick={() => void onRetry(source.id)}><RotateCcw size={13}/></button><button className="icon-button" disabled={actionDisabled} aria-label={`Replace ${source.source}`} title="Choose the file again" onClick={() => { setReplacementId(source.id); if (replacementInput.current) replacementInput.current.accept = '.' + source.source.split('.').pop(); replacementInput.current?.click(); }}><Upload size={13}/></button></>}
        <button className="icon-button" disabled={actionDisabled} aria-label={`${source.status === 'processing' ? 'Cancel processing and remove' : source.status === 'deleting' ? 'Retry deleting' : 'Delete'} ${source.source}`} title={source.status === 'processing' ? 'Cancel processing and remove' : 'Delete source'} onClick={() => { if (source.status === 'deleting') void onDelete(source.id); else setDeleting(source.id); }}>{deletingNow || ownAction === 'deleting' ? <LoaderCircle size={13} className="spin"/> : source.status === 'processing' ? <X size={13}/> : <Trash2 size={13}/>}</button></div>
      {deleting === source.id && <div className="inline-confirm"><span>{source.status === 'processing' ? 'Cancel processing and remove this source?' : 'Delete this source?'}</span><button disabled={actionDisabled} onClick={() => { void onDelete(source.id); setDeleting(null); }}>Remove</button><button onClick={() => setDeleting(null)}>Keep file</button></div>}
    </article>; })}</div>
    <div className="source-panel-footer"><Plus size={15}/><span>Good research starts with good sources.</span></div>
  </aside>;
}

