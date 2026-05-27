const STAGES = [
  {
    id:    'stage1',
    label: 'Stage 1',
    sub:   'Baseline RAG',
    desc:  'Basic retrieval + generation. No verification.',
    icon:  '◈',
  },
  {
    id:    'stage2',
    label: 'Stage 2',
    sub:   'Faithfulness Scored',
    desc:  'Adds faithfulness & confidence scoring with fallback.',
    icon:  '◉',
  },
  {
    id:    'stage4',
    label: 'Stage 4',
    sub:   'Full Agentic Loop',
    desc:  'Self-correcting loop with adaptive retrieval and DistilBERT verification.',
    icon:  '✦',
  },
]

const VIEWS = [
  { id: 'chat',    icon: '◎', label: 'Chat' },
  { id: 'results', icon: '◈', label: 'Evaluation Results' },
]

export default function Sidebar({ view, onViewChange, stage, onStageChange, onClear }) {
  return (
    <aside style={styles.sidebar}>
      <div style={styles.logo}>
        <span style={styles.logoIcon}>⬡</span>
        <div>
          <div style={styles.logoTitle}>HARA</div>
          <div style={styles.logoSub}>Hallucination-Aware Retrieval Agent</div>
        </div>
      </div>

      <div style={styles.section}>
        <div style={styles.sectionLabel}>View</div>
        {VIEWS.map(v => (
          <button
            key={v.id}
            style={{
              ...styles.viewBtn,
              ...(view === v.id ? styles.viewBtnActive : {}),
            }}
            onClick={() => onViewChange(v.id)}
          >
            <span style={{ fontSize: '14px', color: view === v.id ? 'var(--accent-light)' : 'var(--text-muted)' }}>
              {v.icon}
            </span>
            {v.label}
          </button>
        ))}
      </div>

      {view === 'chat' && (
      <div style={styles.section}>
        <div style={styles.sectionLabel}>Pipeline Stage</div>
        {STAGES.map(s => (
          <button
            key={s.id}
            style={{
              ...styles.stageBtn,
              ...(stage === s.id ? styles.stageBtnActive : {}),
            }}
            onClick={() => onStageChange(s.id)}
          >
            <div style={styles.stageBtnTop}>
              <span style={{
                ...styles.stageIcon,
                color: stage === s.id ? 'var(--accent-light)' : 'var(--text-muted)',
              }}>
                {s.icon}
              </span>
              <div>
                <div style={styles.stageLabel}>{s.label}</div>
                <div style={styles.stageSub}>{s.sub}</div>
              </div>
              {stage === s.id && <span style={styles.activeIndicator} />}
            </div>
            {stage === s.id && (
              <div style={styles.stageDesc}>{s.desc}</div>
            )}
          </button>
        ))}
      </div>
      )}

      <div style={styles.section}>
        <div style={styles.sectionLabel}>Model Stack</div>
        <div style={styles.infoCard}>
          <InfoRow label="LLM" value="llama3.2 (Ollama)" />
          <InfoRow label="Embedder" value="all-MiniLM-L6-v2" />
          <InfoRow label="Retrieval" value="FAISS (cosine)" />
          <InfoRow label="Verifier" value="DistilBERT" />
          <InfoRow label="Reranker" value="ms-marco-MiniLM" />
        </div>
      </div>

      {view === 'chat' && (
        <div style={styles.footer}>
          <button style={styles.clearBtn} onClick={onClear}>
            Clear conversation
          </button>
        </div>
      )}
    </aside>
  )
}

function InfoRow({ label, value }) {
  return (
    <div style={styles.infoRow}>
      <span style={styles.infoLabel}>{label}</span>
      <span style={styles.infoValue}>{value}</span>
    </div>
  )
}

const styles = {
  sidebar: {
    width:          '280px',
    flexShrink:     0,
    background:     'var(--bg-sidebar)',
    borderRight:    '1px solid var(--border)',
    display:        'flex',
    flexDirection:  'column',
    padding:        '20px 16px',
    gap:            '24px',
    overflowY:      'auto',
  },
  logo: {
    display:    'flex',
    alignItems: 'center',
    gap:        '12px',
    padding:    '4px 0 8px',
  },
  logoIcon: {
    fontSize:   '26px',
    color:      'var(--accent-light)',
    lineHeight: 1,
  },
  logoTitle: {
    fontWeight: 600,
    fontSize:   '15px',
    color:      'var(--text-primary)',
  },
  logoSub: {
    fontSize: '12px',
    color:    'var(--text-muted)',
    marginTop:'2px',
  },
  section: {
    display:       'flex',
    flexDirection: 'column',
    gap:           '8px',
  },
  sectionLabel: {
    fontSize:      '11px',
    fontWeight:    600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color:         'var(--text-muted)',
    marginBottom:  '2px',
  },
  viewBtn: {
    display:       'flex',
    alignItems:    'center',
    gap:           '8px',
    background:    'transparent',
    border:        '1px solid transparent',
    borderRadius:  'var(--radius-sm)',
    padding:       '8px 12px',
    cursor:        'pointer',
    textAlign:     'left',
    color:         'var(--text-secondary)',
    fontSize:      '13px',
    fontWeight:    500,
    width:         '100%',
    transition:    'all 0.15s ease',
  },
  viewBtnActive: {
    background: 'var(--bg-active)',
    border:     '1px solid var(--border-light)',
    color:      'var(--text-primary)',
  },
  stageBtn: {
    background:    'transparent',
    border:        '1px solid transparent',
    borderRadius:  'var(--radius-sm)',
    padding:       '10px 12px',
    cursor:        'pointer',
    textAlign:     'left',
    color:         'var(--text-secondary)',
    transition:    'all 0.15s ease',
    width:         '100%',
  },
  stageBtnActive: {
    background:    'var(--bg-active)',
    border:        '1px solid var(--border-light)',
    color:         'var(--text-primary)',
  },
  stageBtnTop: {
    display:    'flex',
    alignItems: 'center',
    gap:        '10px',
  },
  stageIcon: {
    fontSize:   '18px',
    lineHeight: 1,
    flexShrink: 0,
  },
  stageLabel: {
    fontWeight: 600,
    fontSize:   '13px',
  },
  stageSub: {
    fontSize: '11px',
    color:    'var(--text-muted)',
    marginTop:'1px',
  },
  activeIndicator: {
    marginLeft:  'auto',
    width:       '6px',
    height:      '6px',
    borderRadius:'50%',
    background:  'var(--accent-light)',
    flexShrink:  0,
  },
  stageDesc: {
    fontSize:   '12px',
    color:      'var(--text-muted)',
    marginTop:  '8px',
    lineHeight: 1.5,
  },
  infoCard: {
    background:   'var(--bg-primary)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    padding:      '10px 12px',
    display:      'flex',
    flexDirection:'column',
    gap:          '6px',
  },
  infoRow: {
    display:        'flex',
    justifyContent: 'space-between',
    alignItems:     'center',
    gap:            '8px',
  },
  infoLabel: {
    fontSize: '12px',
    color:    'var(--text-muted)',
  },
  infoValue: {
    fontSize:  '12px',
    color:     'var(--text-secondary)',
    fontWeight: 500,
    textAlign: 'right',
  },
  footer: {
    marginTop: 'auto',
  },
  clearBtn: {
    width:        '100%',
    padding:      '9px 12px',
    background:   'transparent',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color:        'var(--text-muted)',
    fontSize:     '13px',
    cursor:       'pointer',
    transition:   'all 0.15s ease',
  },
}
