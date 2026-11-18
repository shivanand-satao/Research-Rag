import { Link } from 'react-router-dom';

export function Brand({ compact = false }: { compact?: boolean }) {
  return <Link to="/" className="brand" aria-label="Research RAG home">
    <span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span>
    {!compact && <span>Research<span className="brand-suffix"> RAG</span></span>}
  </Link>;
}

