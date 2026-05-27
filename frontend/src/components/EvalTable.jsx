import { useState, useEffect } from 'react'

const STAGES = ['stage1', 'stage2', 'stage3', 'stage4']
const STAGE_LABELS = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
const STAGE_SUBS = ['Baseline RAG', 'Faithfulness Scored', 'Adaptive Retrieval', 'Full Agentic Loop']

const METRICS = [
  { key: 'exact_match',        label: 'Exact Match',        unit: '%',  higherBetter: true,  fmt: v => `${v.toFixed(1)}%` },
  { key: 'precision',          label: 'Precision',          unit: '',   higherBetter: true,  fmt: v => v.toFixed(4) },
  { key: 'recall',             label: 'Recall',             unit: '',   higherBetter: true,  fmt: v => v.toFixed(4) },
  { key: 'f1',                 label: 'F1 Score',           unit: '',   higherBetter: true,  fmt: v => v.toFixed(4) },
  { key: 'macro_f1',           label: 'Macro F1',           unit: '',   higherBetter: true,  fmt: v => v === null ? null : v.toFixed(4) },
  { key: 'hallucination_rate', label: 'Hallucination Rate', unit: '%',  higherBetter: false, fmt: v => `${v.toFixed(1)}%` },
  { key: 'abstention_rate',    label: 'Abstention Rate',    unit: '%',  higherBetter: false, fmt: v => `${v.toFixed(1)}%` },
]

function cellColor(val, metric, allVals) {
  if (val === null || val === undefined) return 'transparent'
  const nums = allVals.filter(v => v !== null && v !== undefined)
  if (nums.length < 2) return 'transparent'
  const min = Math.min(...nums)
  const max = Math.max(...nums)
  if (max === min) return 'transparent'
  const ratio = (val - min) / (max - min)
  const intensity = Math.round(ratio * 40)
  return metric.higherBetter
    ? `rgba(34, 197, 94, ${intensity / 100})`
    : `rgba(239, 68, 68, ${(1 - ratio) * 0.4})`
}

export default function EvalTable() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => { loadResults() }, [])

  async function loadResults() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/results')
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Failed to load results')
      }
      setData(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <LoadingState />
  if (error)   return <ErrorState message={error} onRetry={loadResults} />

  const { summary, num_samples, timestamp, elapsed_min } = data

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>Pipeline Evaluation Results</h1>
          <p style={styles.subtitle}>
            {num_samples} HotpotQA validation samples · {elapsed_min} min ·{' '}
            {new Date(timestamp).toLocaleString()}
          </p>
        </div>
        <button style={styles.refreshBtn} onClick={loadResults}>
          ↻ Refresh
        </button>
      </div>

      {/* Stage header cards */}
      <div style={styles.stageCards}>
        {STAGES.map((s, i) => {
          const d = summary[s]
          return (
            <div key={s} style={styles.stageCard}>
              <div style={styles.stageCardNum}>Stage {i + 1}</div>
              <div style={styles.stageCardName}>{STAGE_SUBS[i]}</div>
              <div style={styles.stageCardF1}>
                F1 {d.f1.toFixed(3)}
              </div>
            </div>
          )
        })}
      </div>

      {/* Main metrics table */}
      <div style={styles.tableWrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={{ ...styles.th, ...styles.metricCol }}>Metric</th>
              {STAGE_LABELS.map(l => (
                <th key={l} style={styles.th}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m, ri) => {
              const allVals = STAGES.map(s => summary[s][m.key])
              return (
                <tr key={m.key} style={ri % 2 === 0 ? styles.rowEven : styles.rowOdd}>
                  <td style={{ ...styles.td, ...styles.metricCol }}>
                    <span style={styles.metricLabel}>{m.label}</span>
                  </td>
                  {STAGES.map(s => {
                    const val = summary[s][m.key]
                    const bg  = cellColor(val, m, allVals)
                    const isNA = val === null || val === undefined
                    return (
                      <td key={s} style={{ ...styles.td, background: bg }}>
                        {isNA
                          ? <span style={styles.naCell}>N/A</span>
                          : <span style={styles.valCell}>{m.fmt(val)}</span>
                        }
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Per-type F1 breakdown */}
      <div style={styles.breakdowns}>
        {['stage3', 'stage4'].map((s, i) => {
          const per = summary[s]?.per_type_f1 || {}
          if (!Object.keys(per).length) return null
          return (
            <div key={s} style={styles.breakdown}>
              <div style={styles.breakdownTitle}>
                {i === 0 ? 'Stage 3' : 'Stage 4'} — F1 by Query Type
              </div>
              <div style={styles.typeList}>
                {Object.entries(per).map(([qt, v]) => (
                  <div key={qt} style={styles.typeRow}>
                    <span style={{
                      ...styles.typeChip,
                      background: qt === 'SIMPLE'     ? 'rgba(96,165,250,0.12)'
                                : qt === 'MULTI_HOP'  ? 'rgba(167,139,250,0.12)'
                                : 'rgba(52,211,153,0.12)',
                      color:      qt === 'SIMPLE'     ? '#60a5fa'
                                : qt === 'MULTI_HOP'  ? '#a78bfa'
                                : '#34d399',
                      border: `1px solid ${
                        qt === 'SIMPLE'    ? 'rgba(96,165,250,0.3)'
                      : qt === 'MULTI_HOP' ? 'rgba(167,139,250,0.3)'
                      : 'rgba(52,211,153,0.3)'
                      }`,
                    }}>
                      {qt}
                    </span>
                    <div style={styles.typeBar}>
                      <div style={{
                        ...styles.typeBarFill,
                        width: `${v * 100}%`,
                        background: qt === 'SIMPLE'    ? '#60a5fa'
                                  : qt === 'MULTI_HOP' ? '#a78bfa'
                                  : '#34d399',
                      }} />
                    </div>
                    <span style={styles.typeVal}>{v.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>

      <div style={styles.note}>
        To regenerate results: <code style={styles.code}>python Stage_6_Evaluation.py --samples 100</code>
      </div>
    </div>
  )
}

function LoadingState() {
  return (
    <div style={styles.centered}>
      <div style={styles.spinner} />
      <p style={{ color: 'var(--text-muted)', marginTop: '16px' }}>Loading results…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

function ErrorState({ message, onRetry }) {
  const isNotFound = message.includes('No evaluation results')
  return (
    <div style={styles.centered}>
      <div style={styles.errorBox}>
        <div style={styles.errorIcon}>{isNotFound ? '📊' : '⚠'}</div>
        <div style={styles.errorTitle}>
          {isNotFound ? 'No Results Yet' : 'Failed to Load Results'}
        </div>
        <div style={styles.errorMsg}>{message}</div>
        {isNotFound && (
          <div style={styles.errorCmd}>
            <code>python Stage_6_Evaluation.py --samples 50</code>
          </div>
        )}
        <button style={styles.retryBtn} onClick={onRetry}>Try again</button>
      </div>
    </div>
  )
}

const styles = {
  page: {
    flex:          1,
    overflowY:     'auto',
    padding:       '32px 32px 48px',
    display:       'flex',
    flexDirection: 'column',
    gap:           '24px',
  },
  header: {
    display:        'flex',
    justifyContent: 'space-between',
    alignItems:     'flex-start',
  },
  title: {
    fontSize:   '22px',
    fontWeight: 600,
    color:      'var(--text-primary)',
  },
  subtitle: {
    fontSize:  '13px',
    color:     'var(--text-muted)',
    marginTop: '4px',
  },
  refreshBtn: {
    padding:      '8px 16px',
    background:   'var(--bg-active)',
    border:       '1px solid var(--border-light)',
    borderRadius: 'var(--radius-sm)',
    color:        'var(--text-secondary)',
    fontSize:     '13px',
    cursor:       'pointer',
  },
  stageCards: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap:     '12px',
  },
  stageCard: {
    background:   'var(--bg-message-ai)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding:      '16px',
  },
  stageCardNum: {
    fontSize:   '11px',
    fontWeight: 600,
    color:      'var(--accent-light)',
    letterSpacing:'0.07em',
    textTransform:'uppercase',
  },
  stageCardName: {
    fontSize:  '13px',
    color:     'var(--text-secondary)',
    marginTop: '4px',
  },
  stageCardF1: {
    fontSize:   '22px',
    fontWeight: 700,
    color:      'var(--text-primary)',
    marginTop:  '10px',
  },
  tableWrapper: {
    overflowX:    'auto',
    borderRadius: 'var(--radius)',
    border:       '1px solid var(--border)',
  },
  table: {
    width:          '100%',
    borderCollapse: 'collapse',
    fontSize:       '14px',
  },
  th: {
    padding:         '12px 20px',
    textAlign:       'center',
    fontWeight:      600,
    fontSize:        '13px',
    color:           'var(--text-secondary)',
    background:      'var(--bg-sidebar)',
    borderBottom:    '1px solid var(--border)',
    whiteSpace:      'nowrap',
  },
  metricCol: {
    textAlign: 'left',
    minWidth:  '180px',
  },
  td: {
    padding:      '11px 20px',
    textAlign:    'center',
    borderBottom: '1px solid var(--border)',
    transition:   'background 0.2s',
  },
  rowEven: { background: 'var(--bg-primary)' },
  rowOdd:  { background: 'rgba(255,255,255,0.015)' },
  metricLabel: {
    color:      'var(--text-primary)',
    fontWeight: 500,
  },
  valCell: {
    color:      'var(--text-primary)',
    fontWeight: 500,
    fontVariantNumeric: 'tabular-nums',
  },
  naCell: {
    color:     'var(--text-muted)',
    fontSize:  '12px',
    fontStyle: 'italic',
  },
  breakdowns: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap:     '16px',
  },
  breakdown: {
    background:   'var(--bg-message-ai)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding:      '18px 20px',
  },
  breakdownTitle: {
    fontSize:   '13px',
    fontWeight: 600,
    color:      'var(--text-secondary)',
    marginBottom:'14px',
  },
  typeList: {
    display:       'flex',
    flexDirection: 'column',
    gap:           '10px',
  },
  typeRow: {
    display:    'flex',
    alignItems: 'center',
    gap:        '10px',
  },
  typeChip: {
    padding:      '2px 10px',
    borderRadius: '20px',
    fontSize:     '11px',
    fontWeight:   600,
    flexShrink:   0,
    width:        '100px',
    textAlign:    'center',
  },
  typeBar: {
    flex:         1,
    height:       '6px',
    background:   'var(--bg-active)',
    borderRadius: '3px',
    overflow:     'hidden',
  },
  typeBarFill: {
    height:       '100%',
    borderRadius: '3px',
    transition:   'width 0.5s ease',
  },
  typeVal: {
    fontSize:           '13px',
    color:              'var(--text-primary)',
    fontWeight:         500,
    fontVariantNumeric: 'tabular-nums',
    flexShrink:         0,
    width:              '52px',
    textAlign:          'right',
  },
  note: {
    fontSize: '13px',
    color:    'var(--text-muted)',
  },
  code: {
    background:   'var(--bg-active)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius-xs)',
    padding:      '2px 8px',
    fontFamily:   'monospace',
    fontSize:     '12px',
    color:        'var(--text-secondary)',
  },
  centered: {
    flex:           1,
    display:        'flex',
    flexDirection:  'column',
    alignItems:     'center',
    justifyContent: 'center',
    padding:        '48px',
  },
  spinner: {
    width:       '36px',
    height:      '36px',
    border:      '3px solid var(--border-light)',
    borderTopColor: 'var(--accent-light)',
    borderRadius:'50%',
    animation:   'spin 0.8s linear infinite',
  },
  errorBox: {
    background:   'var(--bg-message-ai)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding:      '32px 40px',
    textAlign:    'center',
    maxWidth:     '480px',
    display:      'flex',
    flexDirection:'column',
    gap:          '12px',
    alignItems:   'center',
  },
  errorIcon:  { fontSize: '36px' },
  errorTitle: { fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' },
  errorMsg:   { fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.6 },
  errorCmd: {
    background:   'var(--bg-primary)',
    border:       '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    padding:      '10px 16px',
    fontFamily:   'monospace',
    fontSize:     '13px',
    color:        'var(--accent-light)',
  },
  retryBtn: {
    marginTop:    '4px',
    padding:      '8px 20px',
    background:   'var(--accent)',
    border:       'none',
    borderRadius: 'var(--radius-sm)',
    color:        '#fff',
    fontSize:     '13px',
    cursor:       'pointer',
  },
}
