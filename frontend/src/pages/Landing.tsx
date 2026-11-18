import { ArrowUpRight, Check, FileText, Focus, GitBranch, MoveRight, ScanText } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Brand } from '../components/Brand';
import { useAuth } from '../lib/auth';

export function Landing() {
  const { session } = useAuth();
  const start = session ? '/app' : '/signup';
  return <div className="landing">
    <header className="landing-nav"><Brand/><nav aria-label="Main navigation"><a href="#method">The method</a><a href="#workspace">Workspace</a><Link to={session ? '/app' : '/login'}>{session ? 'Open workspace' : 'Log in'} <ArrowUpRight size={15}/></Link></nav></header>
    <main>
      <section className="hero">
        <div className="hero-copy reveal"><span className="eyebrow"><span className="status-dot"/> YOUR SOURCES. A CLEARER PICTURE.</span>
          <h1>Less searching.<br/>More <em>understanding.</em></h1>
          <p>Turn a stack of papers into a conversation.<br/>Find the connections. Question the evidence.<br/>Trace every answer back to the source.</p>
          <div className="hero-actions"><Link className="button primary" to={start}>Start your research <ArrowUpRight size={18}/></Link><a className="text-link" href="#method">See how it works <MoveRight size={17}/></a></div>
          <div className="hero-note"><span>PDF</span><span>DOCX</span><span>TXT</span><span>MD</span><i/> One connected workspace.</div>
        </div>
        <div className="evidence-art reveal" aria-label="Illustration of research sources connected to a cited answer">
          <div className="art-grid"/><span className="art-coordinate top">RESEARCH FIELD / 001</span><span className="art-coordinate bottom">FROM INFORMATION TO INSIGHT</span>
          <svg className="art-connections" viewBox="0 0 500 460" aria-hidden="true"><path d="M125 120 C250 120 170 250 285 250 M130 325 C220 325 200 250 285 250 M380 80 C380 180 285 180 285 250"/><circle cx="285" cy="250" r="100"/><circle cx="285" cy="250" r="145"/></svg>
          <div className="paper-card paper-one"><FileText size={19}/><span>01 / SOURCE</span><strong>The original evidence</strong><div className="paper-lines"><i/><i/><i/></div><small>Methods · Findings · Context</small></div>
          <div className="paper-card paper-two"><FileText size={19}/><span>02 / SOURCE</span><strong>A different perspective</strong><div className="paper-lines"><i/><i/></div></div>
          <div className="art-node"><GitBranch size={20}/></div>
          <div className="insight-card"><span><span className="status-dot"/> CONNECTION FOUND</span><strong>Ideas become clearer<br/>when evidence connects.</strong><div><span className="citation-tag">[1]</span><span className="citation-tag">[2]</span><small>Grounded in your sources</small></div></div>
        </div>
      </section>
      <div className="principle-strip"><span>A WORKSPACE FOR CURIOUS MINDS</span><p>Your documents. Your questions. <strong>Your next discovery.</strong></p><Focus size={24}/></div>
      <section id="method" className="method-section"><div className="section-heading"><span className="eyebrow">01 / THE METHOD</span><h2>Follow the thread.<br/><span>Keep the evidence.</span></h2><p>Research moves forward when you can see where an answer comes from.</p></div>
        <div className="method-grid">{[
          { icon: FileText, number: '01', title: 'Bring your sources', text: 'Upload papers, reports, notes, and working documents. Build a library that belongs to you.' },
          { icon: ScanText, number: '02', title: 'Ask the better question', text: 'Explore a single source or compare several. Follow up naturally without losing the thread.' },
          { icon: GitBranch, number: '03', title: 'Make the connection', text: 'Read answers with source citations. Inspect the supporting passage and return to the original.' },
        ].map(({ icon: Icon, number, title, text }) => <article key={number}><div className="method-top"><Icon size={24}/><span>{number}</span></div><h3>{title}</h3><p>{text}</p></article>)}</div>
      </section>
      <section id="workspace" className="workspace-promo"><div><span className="eyebrow">02 / ROOM TO THINK</span><h2>A little less noise.<br/>A lot more <em>focus.</em></h2><p>A calm space for the work that matters. Pick up a conversation exactly where you left off.</p><Link className="button primary" to={start}>Find your next insight <ArrowUpRight size={18}/></Link></div><ul>{['A private library for every account', 'Conversations that remember the context', 'Saved history, ready when you are', 'Source passages alongside each answer'].map(item => <li key={item}><Check size={18}/>{item}</li>)}</ul></section>
    </main>
    <footer className="landing-footer"><Brand/><span>Built for the pursuit of understanding.</span><a href="#">Back to top ↑</a></footer>
  </div>;
}

