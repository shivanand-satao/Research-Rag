import { lazy, Suspense, useState } from 'react';
import type { FormEvent } from 'react';
import { Navigate, Route, Routes, useSearchParams } from 'react-router-dom';
import { LoaderCircle } from 'lucide-react';
import { useAuth } from './lib/auth';
import { Landing } from './pages/Landing';
import { AuthPage } from './pages/AuthPage';
import { errorMessage } from './lib/api';

const Workspace = lazy(() => import('./pages/Workspace').then(module => ({ default: module.Workspace })));
const DocumentViewer = lazy(() => import('./pages/DocumentViewer').then(module => ({ default: module.DocumentViewer })));

function ProtectedWorkspace() {
  const { loading, session, error } = useAuth();
  if (loading) return <div className="page-status"><LoaderCircle className="spin"/> Opening your workspace…</div>;
  if (error) return <div className="page-status"><p role="alert">{error}</p><button className="button secondary" onClick={() => location.reload()}>Retry connection</button></div>;
  return session ? <Suspense fallback={<div className="page-status"><LoaderCircle className="spin"/> Opening your workspace…</div>}><Workspace key={session.user.id}/></Suspense> : <Navigate to="/login" replace/>;
}

function Callback() {
  const { loading, session, client, error } = useAuth();
  const [params] = useSearchParams();
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  async function update(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setNotice('');
    try {
      const password = String(new FormData(event.currentTarget).get('password'));
      const { error } = await client!.auth.updateUser({ password });
      if (error) throw error;
      setDone(true);
    } catch (error) { setNotice(errorMessage(error)); } finally { setBusy(false); }
  }
  if (loading) return <div className="page-status"><LoaderCircle className="spin"/> Confirming your account…</div>;
  if (params.get('error_description') || error) return <div className="page-status"><p role="alert">{params.get('error_description') || error}</p><a href="/login" className="button secondary">Return to log in</a></div>;
  if (params.get('recovery') && session && !done) return <div className="page-status"><form className="recovery-form" onSubmit={update}><h2>Choose a new password.</h2><label>New password<input name="password" type="password" autoComplete="new-password" minLength={8} required/></label>{notice && <p role="alert">{notice}</p>}<button disabled={busy} className="button primary">Save password</button></form></div>;
  return <Navigate to={session ? '/app' : '/login'} replace/>;
}

export function App() {
  return <Routes><Route path="/" element={<Landing/>}/><Route path="/login" element={<AuthPage key="login" mode="login"/>}/><Route path="/signup" element={<AuthPage key="signup" mode="signup"/>}/><Route path="/reset-password" element={<AuthPage key="reset" mode="reset"/>}/><Route path="/auth/callback" element={<Callback/>}/><Route path="/app" element={<ProtectedWorkspace/>}/><Route path="/documents/:documentId" element={<Suspense fallback={<div className="page-status">Opening document…</div>}><DocumentViewer/></Suspense>}/><Route path="*" element={<Navigate to="/" replace/>}/></Routes>;
}
