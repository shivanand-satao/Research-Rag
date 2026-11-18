import { useState } from 'react';
import type { FormEvent } from 'react';
import { ArrowLeft, ArrowUpRight, Eye, EyeOff, LoaderCircle, Mail } from 'lucide-react';
import { Link, Navigate } from 'react-router-dom';
import { Brand } from '../components/Brand';
import { useAuth } from '../lib/auth';
import { errorMessage } from '../lib/api';

export function AuthPage({ mode }: { mode: 'login' | 'signup' | 'reset' }) {
  const { client, session, loading, error: connectionError, redirectUrl } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [visible, setVisible] = useState(false);
  if (session && mode !== 'reset') return <Navigate to="/app" replace/>;
  const signup = mode === 'signup';
  const reset = mode === 'reset';
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!client) return;
    const values = new FormData(event.currentTarget);
    const email = String(values.get('email')).trim();
    const password = String(values.get('password'));
    setBusy(true); setError(''); setNotice('');
    try {
      if (reset) {
        const { error } = await client.auth.resetPasswordForEmail(email, { redirectTo: redirectUrl + '?recovery=1' });
        if (error) throw error;
        setNotice('If an account exists for this email, a password reset link is on its way.');
      } else if (signup) {
        const { data, error } = await client.auth.signUp({ email, password,
          options: { data: { display_name: String(values.get('name')).trim() }, emailRedirectTo: redirectUrl } });
        if (error) throw error;
        if (!data.session) setNotice('Check your email to confirm your account, then return here to log in.');
      } else {
        const { error } = await client.auth.signInWithPassword({ email, password });
        if (error) throw error;
      }
    } catch (error) { setError(errorMessage(error)); } finally { setBusy(false); }
  }
  return <main className="auth-layout"><aside className="auth-story"><Brand/><div><span className="eyebrow">THE NEXT DISCOVERY STARTS HERE</span><h1>Good questions.<br/><em>Grounded answers.</em></h1><p>A private space to connect your sources<br/>and develop your thinking.</p><div className="auth-quote"><span>“</span>Understanding begins<br/>with a closer look.<small>RESEARCH RAG</small></div></div><Link to="/" className="text-link"><ArrowLeft size={16}/> Back to the overview</Link></aside>
    <section className="auth-form-side"><div className="mobile-brand"><Brand/></div><div className="auth-form-wrap"><span className="eyebrow">YOUR RESEARCH WORKSPACE</span><h2>{reset ? 'Reset your password.' : signup ? 'Make room for discovery.' : 'Welcome back.'}</h2><p>{reset ? 'We’ll send you a link to choose a new password.' : signup ? 'Create an account. Start connecting the dots.' : 'Your sources and conversations are waiting.'}</p>
      {(error || connectionError) && <div className="alert error" role="alert">{error || connectionError}</div>}
      {notice && <div className="alert success" role="status"><Mail size={18}/>{notice}</div>}
      <form onSubmit={submit}>
        {signup && <label>Your name<input name="name" autoComplete="name" placeholder="Alex Morgan" required maxLength={80}/></label>}
        <label>Email address<input name="email" type="email" autoComplete="email" placeholder="you@example.com" required/></label>
        {!reset && <label><span className="field-label">Password{!signup && <Link to="/reset-password">Forgot password?</Link>}</span><div className="password-field"><input name="password" type={visible ? 'text' : 'password'} autoComplete={signup ? 'new-password' : 'current-password'} placeholder={signup ? 'At least 8 characters' : 'Your password'} minLength={signup ? 8 : 1} required/><button type="button" className="icon-button" onClick={() => setVisible(!visible)} aria-label={visible ? 'Hide password' : 'Show password'}>{visible ? <EyeOff size={18}/> : <Eye size={18}/>}</button></div></label>}
        <button className="button primary full-width" disabled={busy || loading || !client}>{busy ? <LoaderCircle className="spin" size={18}/> : <>{reset ? 'Send reset link' : signup ? 'Create account' : 'Enter workspace'}<ArrowUpRight size={18}/></>}</button>
      </form>
      <p className="auth-switch">{reset ? <Link to="/login">Return to log in</Link> : signup ? <>Already have an account? <Link to="/login">Log in</Link></> : <>New here? <Link to="/signup">Create an account</Link></>}</p>
      <div className="auth-bottom">Your library is private to your account.</div>
    </div></section></main>;
}

