import { useState } from 'react';
import { Check, Copy, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { documentViewerUrl } from '../lib/api';
import type { Message as ChatMessage } from '../lib/api';

interface MarkdownNode { type: string; value?: string; url?: string; children?: MarkdownNode[] }

export function Message({ message, onError }: { message: ChatMessage; onError: (error: string) => void }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try { await navigator.clipboard.writeText(message.content); setCopied(true); setTimeout(() => setCopied(false), 1800); }
    catch { onError('Copy is unavailable in this browser. Select the answer text to copy it.'); }
  }
  if (message.role === 'user') return <article className="message user-message"><span className="message-label">YOU</span><p>{message.content}</p></article>;
  const citations = message.metadata.citations ?? [];
  function citationLinks() {
    return (tree: MarkdownNode) => {
      function walk(node: MarkdownNode) {
        if (!node.children || ['link', 'code', 'inlineCode'].includes(node.type)) return;
        node.children = node.children.flatMap(child => {
          if (child.type !== 'text' || !child.value) { walk(child); return [child]; }
          const nodes: MarkdownNode[] = [];
          let cursor = 0;
          for (const match of child.value.matchAll(/\[(\d+)\]/g)) {
            const source = citations.find(citation => citation.number === Number(match[1]));
            if (!source) continue;
            if (match.index! > cursor) nodes.push({ type: 'text', value: child.value.slice(cursor, match.index) });
            nodes.push({ type: 'link', url: documentViewerUrl(source.document_id, source.page), children: [{ type: 'text', value: match[0] }] });
            cursor = match.index! + match[0].length;
          }
          if (!nodes.length) return [child];
          if (cursor < child.value.length) nodes.push({ type: 'text', value: child.value.slice(cursor) });
          return nodes;
        });
      }
      walk(tree);
    };
  }
  return <article className="message assistant-message"><div className="message-label"><span className="mini-brand">r.</span> RESEARCH RAG <span className={`grounding-label ${message.metadata.grounded ? 'grounded' : ''}`}>{message.metadata.grounded ? 'SOURCES CITED' : 'LIMITED EVIDENCE'}</span></div>
    <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm, citationLinks]} components={{ a: ({ href, children }) => {
      const number = /^\[?(\d+)\]?$/.exec(String(children));
      const source = number ? citations.find(citation => citation.number === Number(number[1])) : undefined;
      return <a href={source ? documentViewerUrl(source.document_id, source.page) : href} target="_blank" rel="noopener noreferrer">{children}</a>;
    } }}>{message.content}</ReactMarkdown></div>
    {!!citations.length && <div className="citation-list">{citations.map(source => <a className="citation-chip" key={source.number} href={documentViewerUrl(source.document_id, source.page)} target="_blank" rel="noopener noreferrer" aria-label={`View ${source.source}${source.page ? ` at page ${source.page}` : ''}`}><span>[{source.number}]</span><FileText size={13}/><span>{source.source}</span>{source.page && <small>p. {source.page}</small>}</a>)}</div>}
    <button className="copy-answer" onClick={() => void copy()} aria-label="Copy answer">{copied ? <Check size={14}/> : <Copy size={14}/>} {copied ? 'Copied' : 'Copy answer'}</button>
  </article>;
}

