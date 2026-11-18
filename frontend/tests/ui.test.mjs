import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdir } from 'node:fs/promises';
import { build } from 'esbuild';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';

await mkdir('.cache/ui-tests', { recursive: true });
const fixtures = { name: 'workspace-fixtures', setup(builder) {
  builder.onLoad({ filter: /[\\/]lib[\\/](auth\.tsx|useWorkspace\.ts)$/ }, args => ({
    contents: args.path.endsWith('auth.tsx')
      ? 'export const useAuth = () => globalThis.authFixture;'
      : 'export const useWorkspace = () => globalThis.workspaceFixture;', loader: 'js',
  }));
} };
for (const [name, directory] of [['Message', 'components'], ['SourcePanel', 'components'], ['Workspace', 'pages']]) {
  await build({ entryPoints: [`src/${directory}/${name}.tsx`], bundle: true, platform: 'node', format: 'esm',
    jsx: 'automatic', packages: 'external', outfile: `.cache/ui-tests/${name}.mjs`, plugins: [fixtures] });
}
const { SourcePanel } = await import('../.cache/ui-tests/SourcePanel.mjs');
const { Workspace } = await import('../.cache/ui-tests/Workspace.mjs');
const { Message } = await import('../.cache/ui-tests/Message.mjs');
const noop = () => {};
const ready = { id: 'ready', title: 'ready.pdf', source: 'ready.pdf', status: 'ready', chunk_count: 4, size_bytes: 1000, error: null };
const processing = { ...ready, id: 'processing', title: 'indexing.pdf', source: 'indexing.pdf', status: 'processing',
  activity: { phase: 'Embedding passages', progress: 55 } };
const removing = { ...ready, id: 'removing', title: 'removing.pdf', source: 'removing.pdf', status: 'deleting',
  activity: { phase: 'Removing original', progress: 25 } };
const failed = { ...ready, id: 'failed', title: 'failed.pdf', source: 'failed.pdf', status: 'failed', error: 'Index unavailable' };
const deleteFailed = { ...removing, id: 'delete-failed', title: 'delete-failed.pdf', source: 'delete-failed.pdf',
  error: 'Service unavailable', activity: undefined };
const props = { documents: [ready, processing, removing, failed, deleteFailed], selected: ['ready'],
  disabled: false, selectionDisabled: false, uploading: true, fileActions: { removing: 'deleting' },
  onUpload: noop, onToggle: noop, onDelete: noop, onRetry: noop, onClose: noop };
function tag(html, label) {
  const found = html.match(/<(?:button|textarea|select|input)\b[^>]*>/g)?.find(element => element.includes(`aria-label="${label}"`));
  assert.ok(found, `Missing control: ${label}`);
  return found;
}

test('file work leaves upload and unrelated file controls enabled', () => {
  const html = renderToStaticMarkup(React.createElement(SourcePanel, props));
  assert.doesNotMatch(html.match(/<button\b[^>]*class="upload-zone[^>]*>/)[0], /disabled/);
  assert.doesNotMatch(tag(html, 'Deselect ready.pdf'), /disabled/);
  assert.doesNotMatch(tag(html, 'Delete ready.pdf'), /disabled/);
  assert.match(tag(html, 'Select indexing.pdf'), /disabled/);
  assert.doesNotMatch(tag(html, 'Cancel processing and remove indexing.pdf'), /disabled/);
  assert.match(tag(html, 'Retry deleting removing.pdf'), /disabled/);
  assert.doesNotMatch(tag(html, 'Retry indexing failed.pdf'), /disabled/);
  assert.doesNotMatch(tag(html, 'Retry deleting delete-failed.pdf'), /disabled/);
  assert.match(html, /aria-valuenow="55"/);
  assert.match(html, /You can keep working/);
});

test('a retry disables only the affected file actions', () => {
  const html = renderToStaticMarkup(React.createElement(SourcePanel, { ...props, fileActions: { failed: 'retrying' } }));
  assert.match(tag(html, 'Retry indexing failed.pdf'), /disabled/);
  assert.match(tag(html, 'Replace failed.pdf'), /disabled/);
  assert.doesNotMatch(tag(html, 'Delete ready.pdf'), /disabled/);
});

test('a failed upload can retry its retained local file', () => {
  const local = { ...failed, id: 'local:upload', title: 'upload.pdf', source: 'upload.pdf' };
  const html = renderToStaticMarkup(React.createElement(SourcePanel, { ...props, documents: [local] }));
  assert.doesNotMatch(tag(html, 'Retry uploading upload.pdf'), /disabled/);
  assert.doesNotMatch(html, /href="\/documents\/local/);
});

test('uploading and processing leave chat and history available', () => {
  globalThis.authFixture = { session: { user: { user_metadata: {}, email: 'reader@example.com' } } };
  globalThis.workspaceFixture = { ...props, documents: props.documents, readySelected: ['ready'],
    loading: false, busy: false, uploading: true, error: '', pending: '', messages: [], active: null,
    conversations: [{ id: 'chat', title: 'Saved research' }], mode: 'hybrid', scope: 'auto',
    newConversation: noop, openConversation: noop, renameConversation: noop, deleteConversation: noop,
    setError: noop, refresh: noop, setMode: noop, setScope: noop, upload: noop, toggleSource: noop,
    deleteDocument: noop, retryDocument: noop };
  const html = renderToStaticMarkup(React.createElement(MemoryRouter, null, React.createElement(Workspace)));
  assert.doesNotMatch(tag(html, 'Ask a research question'), /disabled/);
  assert.doesNotMatch(tag(html, 'Retrieval mode'), /disabled/);
  assert.doesNotMatch(html.match(/<button\b[^>]*class="button new-research"[^>]*>/)[0], /disabled/);
  assert.doesNotMatch(html.match(/<button\b[^>]*class="history-open"[^>]*>/)[0], /disabled/);
  assert.match(html, /1 source in focus/);
  assert.doesNotMatch(html, /A SPACE TO MAKE CONNECTIONS|Find the key ideas|Connect the perspectives|Look a little closer/);
});

test('citations, chips, and source titles link to the PDF viewer', () => {
  const message = { id: 'message', role: 'assistant', content: 'The finding [1]. Corroboration [2](https://wrong.example).',
    metadata: { grounded: true, citations: [
      { number: 1, document_id: 'doc-one', source: 'report.pdf', page: 3 },
      { number: 2, document_id: 'doc-two', source: 'methods.pdf', page: 7 },
    ] } };
  const html = renderToStaticMarkup(React.createElement(Message, { message, onError: noop }));
  assert.match(html, /href="\/documents\/doc-one\?page=3"/);
  assert.match(html, /href="\/documents\/doc-two\?page=7"/);
  assert.doesNotMatch(html, /wrong\.example|citation-detail/);
  const source = renderToStaticMarkup(React.createElement(SourcePanel, props));
  assert.match(source, /href="\/documents\/ready"[^>]*class="source-title"/);
});
