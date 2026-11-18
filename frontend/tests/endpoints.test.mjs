import assert from 'node:assert/strict';
import { test } from 'node:test';
import { build } from 'esbuild';
import { mkdir } from 'node:fs/promises';

await mkdir('.cache/endpoint-tests', { recursive: true });
const modules = {};
for (const [name, env] of Object.entries({
  local: {}, production: { VITE_BACKEND_URL: 'https://research-rag-ugc0.onrender.com' },
  slash: { VITE_BACKEND_URL: 'https://research-rag-ugc0.onrender.com/api/' },
})) {
  await build({ stdin: { contents: `export { api } from './src/lib/api'; export { apiUrl, BACKEND_URL, responseError } from './src/lib/endpoints';`,
    resolveDir: process.cwd(), loader: 'ts' }, bundle: true, platform: 'node', format: 'esm', packages: 'external',
    define: { 'import.meta.env': JSON.stringify(env) }, outfile: `.cache/endpoint-tests/${name}.mjs` });
  modules[name] = await import(`../.cache/endpoint-tests/${name}.mjs`);
}

test('local default and production URLs include exactly one /api prefix', () => {
  assert.equal(modules.local.apiUrl('/config'), 'http://127.0.0.1:8000/api/config');
  for (const name of ['production', 'slash']) {
    assert.equal(modules[name].apiUrl('/documents/example/preview'), 'https://research-rag-ugc0.onrender.com/api/documents/example/preview');
    assert.equal(modules[name].apiUrl('conversations'), 'https://research-rag-ugc0.onrender.com/api/conversations');
  }
});

test('authenticated API calls use Render and preserve the user token', async context => {
  const calls = [];
  context.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push({ url, init });
    return new Response(JSON.stringify([]), { headers: { 'content-type': 'application/json' } });
  });
  const client = { auth: { getSession: async () => ({ data: { session: { access_token: 'user-test-token' } } }) } };
  await modules.production.api(client, '/documents');
  assert.equal(calls[0].url, 'https://research-rag-ugc0.onrender.com/api/documents');
  assert.equal(calls[0].init.headers.Authorization, 'Bearer user-test-token');
});

test('HTML gateway errors produce an actionable status instead of a JSON parse failure', async context => {
  context.mock.method(globalThis, 'fetch', async () => new Response('<html>Bad Gateway</html>', { status: 502 }));
  const client = { auth: { getSession: async () => ({ data: { session: { access_token: 'user-test-token' } } }) } };
  await assert.rejects(modules.production.api(client, '/documents'), /HTTP 502/);
  assert.match(await modules.production.responseError(new Response('<html>Bad Gateway</html>', { status: 502 })), /HTTP 502/);
});
