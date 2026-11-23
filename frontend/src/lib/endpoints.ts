const configured = import.meta.env?.VITE_BACKEND_URL?.trim() || 'http://127.0.0.1:8000';
const BACKEND_URL = configured.replace(/\/+$/, '').replace(/\/api$/, '');

export const apiUrl = (path: string) => `${BACKEND_URL}/api${path.startsWith('/') ? path : `/${path}`}`;

export async function responseError(response: Response, fallback = 'The server could not complete this request.'): Promise<string> {
  const body = await response.json().catch(() => null);
  return typeof body?.detail === 'string' ? body.detail : `${fallback} (HTTP ${response.status}). Please retry shortly.`;
}
