import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { createClient } from '@supabase/supabase-js';
import type { Session, SupabaseClient } from '@supabase/supabase-js';
import { apiUrl, responseError } from './endpoints';

interface AuthContextValue { client: SupabaseClient | null; session: Session | null; loading: boolean; error: string | null; redirectUrl: string }
const AuthContext = createContext<AuthContextValue>({ client: null, session: null, loading: true, error: null, redirectUrl: '' });
let initialization: Promise<{ client: SupabaseClient; redirectUrl: string }> | null = null;

function initialize() {
  if (!initialization) initialization = fetch(apiUrl('/config')).then(async response => {
    if (!response.ok) throw new Error(await responseError(response, 'Research RAG is temporarily unavailable'));
    const config = await response.json();
    return { client: createClient(config.supabase_url, config.supabase_key,
      { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: 'pkce' } }),
      redirectUrl: config.redirect_url };
  }).catch(error => { initialization = null; throw error; });
  return initialization;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthContextValue>({ client: null, session: null, loading: true, error: null, redirectUrl: '' });
  useEffect(() => {
    let alive = true;
    let unsubscribe: (() => void) | undefined;
    initialize().then(async ({ client, redirectUrl }) => {
      const { data: { subscription } } = client.auth.onAuthStateChange((_event, session) => {
        if (alive) setState({ client, session, redirectUrl, loading: false, error: null });
      });
      unsubscribe = () => subscription.unsubscribe();
      const { data: { session }, error } = await client.auth.getSession();
      if (alive) setState({ client, session, redirectUrl, loading: false, error: error?.message ?? null });
      else unsubscribe();
    }).catch(error => { if (alive) setState(previous => ({ ...previous, loading: false, error: error.message })); });
    return () => { alive = false; unsubscribe?.(); };
  }, []);
  return <AuthContext.Provider value={state}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);

