import { useEffect, useState } from 'react';
import { ArrowLeft, Download, FileText, LoaderCircle } from 'lucide-react';
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom';
import { useAuth } from '../lib/auth';
import { api, errorMessage } from '../lib/api';
import type { DocumentSource } from '../lib/api';
import { apiUrl, responseError } from '../lib/endpoints';

export function DocumentViewer() {
  const { documentId } = useParams();
  const [params] = useSearchParams();
  const auth = useAuth();
  const [source, setSource] = useState<DocumentSource | null>(null);
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const page = Math.max(1, Math.min(500, Number(params.get('page')) || 1));
  useEffect(() => {
    if (!auth.client || !auth.session || !documentId) return;
    const abort = new AbortController();
    let blobUrl = '';
    setLoading(true); setError('');
    Promise.all([
      api<DocumentSource>(auth.client, `/documents/${documentId}`, { signal: abort.signal }),
      fetch(apiUrl(`/documents/${documentId}/preview`), { signal: abort.signal, headers: { Authorization: `Bearer ${auth.session.access_token}` } }).then(async response => {
        if (!response.ok) {
          throw new Error(await responseError(response, 'This document is unavailable'));
        }
        return response.blob();
      }),
    ]).then(([metadata, blob]) => {
      if (abort.signal.aborted) return;
      blobUrl = URL.createObjectURL(blob);
      setSource(metadata); setUrl(blobUrl); setLoading(false);
    }).catch(error => { if (!abort.signal.aborted) { setError(errorMessage(error)); setLoading(false); } });
    return () => { abort.abort(); if (blobUrl) URL.revokeObjectURL(blobUrl); };
  }, [auth.client, auth.session?.access_token, documentId]);
  if (!auth.loading && !auth.session) return <Navigate to="/login" replace/>;
  return <main className="document-viewer"><header className="viewer-toolbar"><Link to="/app" className="viewer-back"><ArrowLeft size={17}/> Workspace</Link><div className="viewer-title"><FileText size={17}/><h1>{source?.title || 'Document viewer'}</h1></div><span className="viewer-page">Page {page}</span>{url && <a className="button secondary" href={url} download={`${source?.title?.replace(/\.[^.]+$/, '') || 'source'}.pdf`}><Download size={15}/> Download PDF</a>}</header>
    {auth.loading || loading ? <div className="page-status"><LoaderCircle className="spin"/> Opening the document…</div> : error ? <div className="page-status"><p role="alert">{error}</p><button className="button secondary" onClick={() => location.reload()}>Retry</button></div> : <iframe className="pdf-frame" title={source?.title || 'Source PDF'} src={`${url}#page=${page}&view=FitH`} />}
  </main>;
}
