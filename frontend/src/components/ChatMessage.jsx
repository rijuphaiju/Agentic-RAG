import { useState } from 'react'

// ── Label colour helpers ──────────────────────────────────────────────────
const LABEL_CFG = {
  SUPPORTED:   { color: '#22c55e', bg: 'rgba(34,197,94,0.10)',  border: 'rgba(34,197,94,0.25)',  icon: '✓', text: 'Supported'   },
  PARTIAL:     { color: '#f59e0b', bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.25)', icon: '~', text: 'Partial'     },
  UNSUPPORTED: { color: '#ef4444', bg: 'rgba(239,68,68,0.10)',  border: 'rgba(239,68,68,0.25)',  icon: '✕', text: 'Unsupported' },
  ABSTAINED:   { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)',border: 'rgba(148,163,184,0.25)',icon: '—', text: 'Abstained'   },
  UNKNOWN:     { color: '#64748b', bg: 'rgba(100,116,139,0.10)',border: 'rgba(100,116,139,0.20)',icon: '?', text: 'Unknown'     },
}

const QTYPE_CFG = {
  SIMPLE:     { color: '#60a5fa', bg: 'rgba(96,165,250,0.10)',   border: 'rgba(96,165,250,0.25)'  },
  MULTI_HOP:  { color: '#a78bfa', bg: 'rgba(167,139,250,0.10)',  border: 'rgba(167,139,250,0.25)' },
  COMPARISON: { color: '#34d399', bg: 'rgba(52,211,153,0.10)',   border: 'rgba(52,211,153,0.25)'  },
}

// ── Derive per-query metric values from API metadata ─────────────────────
function deriveMetrics(stage, meta) {
  if (!meta) return null

  if (stage === 'stage1') {
    return {
      label:        null,
      confidence:   null,
      hallucinated: null,   // Stage 1 has no verifier — hallucination is undetected
      abstained:    false,
      query_type:   null,
      iterations:   null,
    }
  }

  if (stage === 'stage2') {
    const lbl = meta.label || 'UNKNOWN'
    return {
      label:        lbl,
      confidence:   meta.confidence,
      hallucinated: ['PARTIAL', 'UNSUPPORTED'].includes(lbl),
      abstained:    null,   // N/A for Stage 2
      query_type:   null,
      iterations:   null,
    }
  }

  if (stage === 'stage3') {
    const lbl = meta.label || 'UNKNOWN'
    return {
      label:        lbl,
      confidence:   meta.confidence,
      hallucinated: ['PARTIAL', 'UNSUPPORTED'].includes(lbl),
      abstained:    null,   // N/A for Stage 3
      query_type:   meta.query_type,
      iterations:   null,
    }
  }

  if (stage === 'stage4') {
    const status = meta.status || 'UNKNOWN'
    const abstained = status === 'ABSTAINED'
    return {
      label:        status,
      confidence:   meta.verification_scores
                      ? Math.max(...Object.values(meta.verification_scores))
                      : null,
      hallucinated: !abstained && status !== 'SUPPORTED',
      abstained,
      query_type:   meta.query_type,
      iterations:   meta.iterations,
    }
  }

  return null
}

// ── Main exports ──────────────────────────────────────────────────────────
export default function ChatMessage({ message }) {
  return message.role === 'user'
    ? <UserMessage message={message} />
    : <AssistantMessage message={message} />
}

function UserMessage({ message }) {
  return (
    <div style={s.userRow}>
      <div style={s.userBubble}>{message.text}</div>
      <div style={s.userAvatar}>U</div>
    </div>
  )
}

function AssistantMessage({ message }) {
  const [open, setOpen] = useState(false)
  const meta    = message.metadata
  const stage   = message.stage
  const metrics = deriveMetrics(stage, meta)

  const hasDetails = meta && (
    (stage === 'stage4' && meta.iteration_log?.length) ||
    meta.retrieved_titles?.length
  )

  return (
    <div style={s.aiRow}>
      <div style={s.aiAvatar}>⬡</div>
      <div style={s.aiContent}>

        {/* Answer bubble */}
        <div style={s.aiBubble}>
          <p style={s.answerText}>{message.text}</p>
        </div>

        {/* Per-query metrics panel */}
        {metrics && (
          <MetricsPanel
            stage={stage}
            metrics={metrics}
            meta={meta}
            open={open}
            onToggle={hasDetails ? () => setOpen(v => !v) : null}
          />
        )}

        {/* Expandable detail panel */}
        {open && meta && <DetailPanel meta={meta} stage={stage} />}
      </div>
    </div>
  )
}

// ── Metrics panel ─────────────────────────────────────────────────────────
function MetricsPanel({ stage, metrics, meta, open, onToggle }) {
  const rows = buildMetricRows(stage, metrics, meta)

  return (
    <div style={s.metricsBox}>
      <div style={s.metricsHeader}>
        <span style={s.metricsTitle}>Per-Query Metrics</span>
        <span style={s.stageChip}>{stageLabel(stage)}</span>
      </div>

      <div style={s.metricsGrid}>
        {rows.map(({ label, value, valueEl }) => (
          <div key={label} style={s.metricRow}>
            <span style={s.metricName}>{label}</span>
            <span style={s.metricVal}>{valueEl || value}</span>
          </div>
        ))}
      </div>

      {onToggle && (
        <button style={s.detailsBtn} onClick={onToggle}>
          {open ? 'Hide details ▲' : 'View details ▼'}
        </button>
      )}
    </div>
  )
}

function buildMetricRows(stage, metrics, meta) {
  const rows = []

  // Verifier Label — not applicable for Stage 1 (verifier introduced in Stage 2)
  if (stage === 'stage1') {
    rows.push({
      label:   'Verifier Label',
      valueEl: <span style={{ color: 'var(--text-muted)', fontSize: '12px', fontStyle: 'italic' }}>N/A — no verifier in Stage 1</span>,
    })
  } else {
    const lbl = metrics.label || 'UNKNOWN'
    const cfg = LABEL_CFG[lbl] || LABEL_CFG.UNKNOWN
    rows.push({
      label:   'Verifier Label',
      valueEl: (
        <span style={{ ...s.labelBadge, color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
          {cfg.icon} {cfg.text}
        </span>
      ),
    })
  }

  // Confidence
  if (metrics.confidence !== null) {
    const pct = (metrics.confidence * 100).toFixed(1)
    const color = metrics.confidence >= 0.7 ? '#22c55e'
                : metrics.confidence >= 0.45 ? '#f59e0b'
                : '#ef4444'
    rows.push({
      label:   'Confidence',
      valueEl: <span style={{ color, fontWeight: 600 }}>{pct}%</span>,
    })
  }

  // Query Type
  if (metrics.query_type) {
    const cfg = QTYPE_CFG[metrics.query_type] || {}
    rows.push({
      label:   'Query Type',
      valueEl: (
        <span style={{ ...s.labelBadge, color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
          {metrics.query_type}
        </span>
      ),
    })
  }

  // Retrieval Strategy — Stage 3 only
  if (stage === 'stage3' && meta.retrieval_strategy) {
    const cfg = QTYPE_CFG[meta.query_type] || {}
    rows.push({
      label:   'Retrieval Strategy',
      valueEl: (
        <span style={{ color: cfg.color || 'var(--text-secondary)', fontSize: '12px', fontWeight: 500 }}>
          {meta.retrieval_strategy}
        </span>
      ),
    })
  }

  // Passages Retrieved — Stage 3 only
  if (stage === 'stage3' && meta.num_retrieved != null) {
    rows.push({
      label: 'Passages Retrieved',
      valueEl: <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{meta.num_retrieved}</span>,
    })
  }

  // Iterations (Stage 4)
  if (metrics.iterations !== null && metrics.iterations !== undefined) {
    rows.push({
      label:   'Iterations Used',
      valueEl: (
        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>
          {metrics.iterations} / {meta.iteration_log?.length
            ? Math.max(metrics.iterations, meta.iteration_log.length)
            : metrics.iterations}
        </span>
      ),
    })
  }

  // Hallucination probability from verifier scores
  const scores = meta?.scores || meta?.verification_scores || {}
  const hallProb = ((scores.PARTIAL || 0) + (scores.UNSUPPORTED || 0)) * 100
  if (Object.keys(scores).length > 0) {
    const hColor = hallProb >= 50 ? '#ef4444' : hallProb >= 25 ? '#f59e0b' : '#22c55e'
    rows.push({
      label:   'Hallucination Probability',
      valueEl: <span style={{ color: hColor, fontWeight: 600 }}>{hallProb.toFixed(1)}%</span>,
    })
  } else if (metrics.hallucinated === null) {
    rows.push({
      label:   'Hallucination',
      valueEl: <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '12px' }}>Unknown (no verifier)</span>,
    })
  }

  // EM and F1 against gold answer (only when question is from HotpotQA)
  if (meta?.eval) {
    const ev = meta.eval
    rows.push({
      label:   'Exact Match',
      valueEl: <span style={{ color: ev.em === 1 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
        {ev.em === 1 ? '✓ Yes' : '✕ No'}
      </span>,
    })
    rows.push({
      label:   'F1 Score',
      valueEl: <span style={{ color: ev.f1 >= 0.7 ? '#22c55e' : ev.f1 >= 0.3 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
        {(ev.f1 * 100).toFixed(1)}%
      </span>,
    })
    rows.push({
      label:   'Gold Answer',
      valueEl: <span style={{ color: 'var(--text-muted)', fontSize: '12px', fontStyle: 'italic' }}>{ev.gold_answer}</span>,
    })
  }

  // Abstention Rate
  if (stage === 'stage1') {
    rows.push({
      label:   'Abstention',
      valueEl: <span style={{ color: 'var(--text-muted)' }}>0% (fixed — no mechanism to withhold)</span>,
    })
  } else if (stage === 'stage4') {
    rows.push({
      label:   'Abstention',
      valueEl: metrics.abstained
        ? <span style={{ color: '#94a3b8', fontWeight: 600 }}>Yes — insufficient evidence</span>
        : <span style={{ color: 'var(--text-secondary)' }}>No</span>,
    })
  } else {
    rows.push({
      label:   'Abstention',
      valueEl: <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '12px' }}>N/A — no abstention in this stage</span>,
    })
  }

  // Macro F1 note
  if (stage === 'stage1') {
    rows.push({
      label:   'Macro F1',
      valueEl: <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '12px' }}>N/A — no query type classification in Stage 1</span>,
    })
  }

  return rows
}

function stageLabel(stage) {
  return { stage1: 'Stage 1', stage2: 'Stage 2', stage3: 'Stage 3', stage4: 'Stage 4' }[stage] || stage
}

// ── Expandable detail panel ───────────────────────────────────────────────
function DetailPanel({ meta, stage }) {
  return (
    <div style={s.detailPanel}>
      {/* Verifier score breakdown */}
      {(meta.scores || meta.verification_scores) && (
        <div style={s.detailSection}>
          <div style={s.detailTitle}>Verifier Score Breakdown</div>
          <div style={s.scoreGrid}>
            {Object.entries(meta.scores || meta.verification_scores).map(([k, v]) => {
              const cfg = LABEL_CFG[k] || {}
              return (
                <div key={k} style={s.scoreItem}>
                  <span style={{ ...s.scoreLabel, color: cfg.color }}>{k}</span>
                  <span style={s.scoreVal}>{(v * 100).toFixed(1)}%</span>
                  <div style={s.scoreBar}>
                    <div style={{ ...s.scoreBarFill, width: `${v * 100}%`, background: cfg.color || '#64748b' }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Stage 4 iteration log */}
      {stage === 'stage4' && meta.iteration_log?.length > 0 && (
        <div style={s.detailSection}>
          <div style={s.detailTitle}>Iteration Log</div>
          {meta.iteration_log.map((it, i) => {
            const cfg = LABEL_CFG[it.label] || LABEL_CFG.UNKNOWN
            return (
              <div key={i} style={s.iterRow}>
                <span style={s.iterNum}>#{it.iteration}</span>
                <span style={{ ...s.iterBadge, color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
                  {cfg.icon} {it.label}
                </span>
                <span style={s.iterConf}>{(it.confidence * 100).toFixed(1)}%</span>
                <span style={s.iterQ}>{it.query}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Retrieved sources */}
      {meta.retrieved_titles?.length > 0 && (
        <div style={s.detailSection}>
          <div style={s.detailTitle}>Retrieved Sources</div>
          {meta.retrieved_titles.map((t, i) => (
            <div key={i} style={s.sourceRow}>
              <span style={s.sourceNum}>{i + 1}</span>
              <span style={s.sourceTitle}>{t}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────
const s = {
  userRow: {
    display: 'flex', justifyContent: 'flex-end',
    alignItems: 'flex-start', gap: '12px',
  },
  userBubble: {
    maxWidth: '65%', background: 'var(--user-bubble)', color: '#fff',
    borderRadius: 'var(--radius) 0 var(--radius) var(--radius)',
    padding: '12px 16px', fontSize: '15px', lineHeight: 1.6, wordBreak: 'break-word',
  },
  userAvatar: {
    width: '34px', height: '34px', borderRadius: '50%',
    background: 'var(--user-bubble)', display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    fontSize: '13px', fontWeight: 600, color: '#fff', flexShrink: 0,
  },
  aiRow:    { display: 'flex', alignItems: 'flex-start', gap: '12px' },
  aiAvatar: {
    width: '34px', height: '34px', borderRadius: '50%',
    background: 'var(--accent-glow)', border: '1px solid var(--accent)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: '16px', color: 'var(--accent-light)', flexShrink: 0,
  },
  aiContent: {
    flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '8px',
  },
  aiBubble: {
    background: 'var(--bg-message-ai)', border: '1px solid var(--border)',
    borderRadius: '0 var(--radius) var(--radius) var(--radius)', padding: '14px 18px',
  },
  answerText: {
    fontSize: '15px', lineHeight: 1.7, color: 'var(--text-primary)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0,
  },

  // Metrics box
  metricsBox: {
    background: 'var(--bg-primary)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)', padding: '12px 14px',
    display: 'flex', flexDirection: 'column', gap: '10px',
  },
  metricsHeader: {
    display: 'flex', alignItems: 'center', gap: '8px',
  },
  metricsTitle: {
    fontSize: '11px', fontWeight: 600, letterSpacing: '0.07em',
    textTransform: 'uppercase', color: 'var(--text-muted)',
  },
  stageChip: {
    fontSize: '10px', fontWeight: 600, padding: '1px 7px',
    borderRadius: '10px', background: 'var(--bg-active)',
    border: '1px solid var(--border-light)', color: 'var(--accent-light)',
    letterSpacing: '0.05em',
  },
  metricsGrid: {
    display: 'flex', flexDirection: 'column', gap: '5px',
  },
  metricRow: {
    display: 'flex', alignItems: 'center',
    justifyContent: 'space-between', gap: '12px',
    padding: '4px 0',
    borderBottom: '1px solid rgba(255,255,255,0.04)',
  },
  metricName: {
    fontSize: '12px', color: 'var(--text-muted)',
    flexShrink: 0, minWidth: '130px',
  },
  metricVal: {
    fontSize: '12px', color: 'var(--text-secondary)',
    fontWeight: 500, textAlign: 'right',
  },
  labelBadge: {
    display: 'inline-flex', alignItems: 'center', gap: '4px',
    padding: '2px 9px', borderRadius: '20px',
    fontSize: '11px', fontWeight: 600,
  },
  detailsBtn: {
    alignSelf: 'flex-end', background: 'transparent', border: 'none',
    color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer',
    padding: '2px 4px', borderRadius: 'var(--radius-xs)',
  },

  // Detail panel
  detailPanel: {
    background: 'var(--bg-primary)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)', padding: '14px 16px',
    display: 'flex', flexDirection: 'column', gap: '18px',
  },
  detailSection: {
    display: 'flex', flexDirection: 'column', gap: '8px',
  },
  detailTitle: {
    fontSize: '11px', fontWeight: 600, letterSpacing: '0.07em',
    textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '2px',
  },
  scoreGrid: {
    display: 'flex', flexDirection: 'column', gap: '6px',
  },
  scoreItem: {
    display: 'grid', gridTemplateColumns: '110px 48px 1fr',
    alignItems: 'center', gap: '10px',
  },
  scoreLabel: {
    fontSize: '12px', fontWeight: 600,
  },
  scoreVal: {
    fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)',
    textAlign: 'right', fontVariantNumeric: 'tabular-nums',
  },
  scoreBar: {
    height: '5px', background: 'var(--bg-active)',
    borderRadius: '3px', overflow: 'hidden',
  },
  scoreBarFill: {
    height: '100%', borderRadius: '3px', transition: 'width 0.4s ease',
  },
  iterRow: {
    display: 'flex', alignItems: 'center', gap: '8px',
    fontSize: '12px', padding: '4px 0',
    borderBottom: '1px solid var(--border)',
  },
  iterNum:  { color: 'var(--text-muted)', flexShrink: 0, width: '22px' },
  iterBadge: {
    display: 'inline-flex', alignItems: 'center', gap: '3px',
    padding: '1px 8px', borderRadius: '10px',
    fontSize: '11px', fontWeight: 600, flexShrink: 0,
  },
  iterConf: { color: 'var(--text-secondary)', flexShrink: 0, width: '44px' },
  iterQ: {
    color: 'var(--text-muted)', overflow: 'hidden',
    textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
  },
  sourceRow: {
    display: 'flex', alignItems: 'baseline', gap: '8px',
    fontSize: '13px', padding: '3px 0',
  },
  sourceNum:   { color: 'var(--accent-light)', fontWeight: 600, flexShrink: 0, width: '18px' },
  sourceTitle: { color: 'var(--text-secondary)' },
}
