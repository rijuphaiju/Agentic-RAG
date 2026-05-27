import { useState } from 'react'

const STATUS_CONFIG = {
  SUPPORTED:   { color: 'var(--status-supported)', icon: '✓', label: 'Supported' },
  PARTIAL:     { color: 'var(--status-partial)',   icon: '~', label: 'Partial'   },
  BEST_EFFORT: { color: 'var(--status-effort)',    icon: '↻', label: 'Best Effort'},
  ERROR:       { color: 'var(--status-error)',     icon: '✕', label: 'Error'     },
}

const QUERY_TYPE_COLOR = {
  SIMPLE:     '#60a5fa',
  MULTI_HOP:  '#a78bfa',
  COMPARISON: '#34d399',
}

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  return isUser
    ? <UserMessage message={message} />
    : <AssistantMessage message={message} />
}

function UserMessage({ message }) {
  return (
    <div style={styles.userRow}>
      <div style={styles.userBubble}>
        {message.text}
      </div>
      <div style={styles.userAvatar}>U</div>
    </div>
  )
}

function AssistantMessage({ message }) {
  const [open, setOpen] = useState(false)
  const meta = message.metadata

  return (
    <div style={styles.aiRow}>
      <div style={styles.aiAvatar}>⬡</div>
      <div style={styles.aiContent}>
        <div style={styles.aiBubble}>
          <p style={styles.answerText}>{message.text}</p>
        </div>

        {meta && message.stage === 'stage4' && (
          <div style={styles.metaBar}>
            <StatusBadge status={meta.status} />
            <QueryTypeBadge type={meta.query_type} />
            <IterBadge iterations={meta.iterations} />

            <button
              style={styles.detailsBtn}
              onClick={() => setOpen(v => !v)}
            >
              {open ? 'Hide details ▲' : 'View details ▼'}
            </button>
          </div>
        )}

        {meta && message.stage === 'stage2' && (
          <div style={styles.metaBar}>
            <ScoreBadge label="Faithfulness" value={meta.faithfulness} />
            <ScoreBadge label="Confidence"   value={meta.confidence}   />
            {meta.fallback_used && (
              <span style={{ ...styles.badge, background: 'rgba(245,158,11,0.12)', color: '#f59e0b', border:'1px solid rgba(245,158,11,0.25)' }}>
                Fallback used
              </span>
            )}
            <button
              style={styles.detailsBtn}
              onClick={() => setOpen(v => !v)}
            >
              {open ? 'Hide sources ▲' : 'View sources ▼'}
            </button>
          </div>
        )}

        {meta && message.stage === 'stage1' && (
          <div style={styles.metaBar}>
            <button
              style={styles.detailsBtn}
              onClick={() => setOpen(v => !v)}
            >
              {open ? 'Hide sources ▲' : 'View sources ▼'}
            </button>
          </div>
        )}

        {open && meta && <MetaPanel meta={meta} stage={message.stage} />}
      </div>
    </div>
  )
}

function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.BEST_EFFORT
  return (
    <span style={{
      ...styles.badge,
      color:      cfg.color,
      background: `${cfg.color}18`,
      border:     `1px solid ${cfg.color}40`,
    }}>
      {cfg.icon} {cfg.label}
    </span>
  )
}

function QueryTypeBadge({ type }) {
  const color = QUERY_TYPE_COLOR[type] || '#a0a0a0'
  return (
    <span style={{
      ...styles.badge,
      color,
      background: `${color}18`,
      border:     `1px solid ${color}40`,
    }}>
      {type}
    </span>
  )
}

function IterBadge({ iterations }) {
  return (
    <span style={{
      ...styles.badge,
      color:      'var(--text-muted)',
      background: 'var(--bg-primary)',
      border:     '1px solid var(--border)',
    }}>
      {iterations}/3 iter
    </span>
  )
}

function ScoreBadge({ label, value }) {
  const pct = Math.round(value * 100)
  const color = pct >= 70 ? 'var(--status-supported)'
              : pct >= 45 ? 'var(--status-partial)'
              : 'var(--status-error)'
  return (
    <span style={{
      ...styles.badge,
      color,
      background: `${color}18`,
      border:     `1px solid ${color}40`,
    }}>
      {label}: {pct}%
    </span>
  )
}

function MetaPanel({ meta, stage }) {
  return (
    <div style={styles.metaPanel}>
      {stage === 'stage4' && meta.iteration_log && (
        <>
          <div style={styles.panelSection}>
            <div style={styles.panelTitle}>Verification Scores</div>
            <div style={styles.scoreRow}>
              {Object.entries(meta.verification_scores || {}).map(([k, v]) => (
                <div key={k} style={styles.scoreItem}>
                  <span style={styles.scoreLabel}>{k}</span>
                  <span style={styles.scoreVal}>{(v * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div style={styles.panelSection}>
            <div style={styles.panelTitle}>Iteration Log</div>
            {meta.iteration_log.map((it, i) => (
              <div key={i} style={styles.iterRow}>
                <span style={styles.iterNum}>#{it.iteration}</span>
                <span style={styles.iterLabel(it.label)}>{it.label}</span>
                <span style={styles.iterConf}>{(it.confidence * 100).toFixed(1)}%</span>
                <span style={styles.iterQ}>{it.query}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {(stage === 'stage1' || stage === 'stage2') && meta.retrieved_titles && (
        <div style={styles.panelSection}>
          <div style={styles.panelTitle}>Retrieved Sources</div>
          {meta.retrieved_titles.map((t, i) => (
            <div key={i} style={styles.sourceRow}>
              <span style={styles.sourceNum}>{i + 1}</span>
              <span style={styles.sourceTitle}>{t}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const styles = {
  userRow: {
    display:        'flex',
    justifyContent: 'flex-end',
    alignItems:     'flex-start',
    gap:            '12px',
  },
  userBubble: {
    maxWidth:     '65%',
    background:   'var(--user-bubble)',
    color:        '#fff',
    borderRadius: 'var(--radius) 0 var(--radius) var(--radius)',
    padding:      '12px 16px',
    fontSize:     '15px',
    lineHeight:   1.6,
    wordBreak:    'break-word',
  },
  userAvatar: {
    width:          '34px',
    height:         '34px',
    borderRadius:   '50%',
    background:     'var(--user-bubble)',
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'center',
    fontSize:       '13px',
    fontWeight:     600,
    color:          '#fff',
    flexShrink:     0,
  },
  aiRow: {
    display:    'flex',
    alignItems: 'flex-start',
    gap:        '12px',
  },
  aiAvatar: {
    width:          '34px',
    height:         '34px',
    borderRadius:   '50%',
    background:     'var(--accent-glow)',
    border:         '1px solid var(--accent)',
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'center',
    fontSize:       '16px',
    color:          'var(--accent-light)',
    flexShrink:     0,
  },
  aiContent: {
    flex:          1,
    minWidth:      0,
    display:       'flex',
    flexDirection: 'column',
    gap:           '8px',
  },
  aiBubble: {
    background:   'var(--bg-message-ai)',
    border:       '1px solid var(--border)',
    borderRadius: '0 var(--radius) var(--radius) var(--radius)',
    padding:      '14px 18px',
  },
  answerText: {
    fontSize:   '15px',
    lineHeight: 1.7,
    color:      'var(--text-primary)',
    whiteSpace: 'pre-wrap',
    wordBreak:  'break-word',
  },
  metaBar: {
    display:    'flex',
    flexWrap:   'wrap',
    alignItems: 'center',
    gap:        '6px',
    paddingLeft:'2px',
  },
  badge: {
    display:      'inline-flex',
    alignItems:   'center',
    gap:          '4px',
    padding:      '3px 9px',
    borderRadius: '20px',
    fontSize:     '12px',
    fontWeight:   500,
    letterSpacing:'0.01em',
  },
  detailsBtn: {
    marginLeft:  'auto',
    background:  'transparent',
    border:      'none',
    color:       'var(--text-muted)',
    fontSize:    '12px',
    cursor:      'pointer',
    padding:     '3px 6px',
    borderRadius:'var(--radius-xs)',
    transition:  'color 0.15s',
  },
  metaPanel: {
    background:   'var(--bg-primary)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    padding:      '14px 16px',
    display:      'flex',
    flexDirection:'column',
    gap:          '16px',
  },
  panelSection: {
    display:       'flex',
    flexDirection: 'column',
    gap:           '8px',
  },
  panelTitle: {
    fontSize:      '11px',
    fontWeight:    600,
    letterSpacing: '0.07em',
    textTransform: 'uppercase',
    color:         'var(--text-muted)',
    marginBottom:  '2px',
  },
  scoreRow: {
    display: 'flex',
    gap:     '16px',
    flexWrap:'wrap',
  },
  scoreItem: {
    display:       'flex',
    flexDirection: 'column',
    gap:           '2px',
  },
  scoreLabel: {
    fontSize: '11px',
    color:    'var(--text-muted)',
  },
  scoreVal: {
    fontSize:   '16px',
    fontWeight: 600,
    color:      'var(--text-primary)',
  },
  iterRow: {
    display:    'flex',
    alignItems: 'center',
    gap:        '8px',
    fontSize:   '12px',
    padding:    '4px 0',
    borderBottom:'1px solid var(--border)',
  },
  iterNum: {
    color:     'var(--text-muted)',
    flexShrink:0,
    width:     '20px',
  },
  iterLabel: (label) => ({
    padding:      '1px 8px',
    borderRadius: '10px',
    fontSize:     '11px',
    fontWeight:   600,
    flexShrink:   0,
    color:        label === 'SUPPORTED' ? 'var(--status-supported)'
                : label === 'PARTIAL'   ? 'var(--status-partial)'
                : 'var(--text-muted)',
    background:   label === 'SUPPORTED' ? 'rgba(34,197,94,0.1)'
                : label === 'PARTIAL'   ? 'rgba(245,158,11,0.1)'
                : 'var(--bg-active)',
  }),
  iterConf: {
    color:     'var(--text-secondary)',
    flexShrink:0,
    width:     '44px',
  },
  iterQ: {
    color:    'var(--text-muted)',
    overflow: 'hidden',
    textOverflow:'ellipsis',
    whiteSpace: 'nowrap',
    flex:     1,
  },
  sourceRow: {
    display:    'flex',
    alignItems: 'baseline',
    gap:        '8px',
    fontSize:   '13px',
    padding:    '3px 0',
  },
  sourceNum: {
    color:     'var(--accent-light)',
    fontWeight:600,
    flexShrink:0,
    width:     '18px',
  },
  sourceTitle: {
    color: 'var(--text-secondary)',
  },
}
