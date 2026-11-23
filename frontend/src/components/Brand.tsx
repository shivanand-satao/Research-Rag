import { Link } from 'react-router-dom';

export function Brand() {
  return <Link to="/" className="brand" aria-label="Research RAG home">
    <span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span>
    <span>Research<span className="brand-suffix"> RAG</span></span>
  </Link>;
}

