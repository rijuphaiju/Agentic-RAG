import ChatMessage from './ChatMessage'

export default function ChatWindow({ messages, loading, bottomRef, sessionStats }) {
  const { total, hallucinated, abstained } = sessionStats || {}
  const hallucinationRate = total > 0 ? ((hallucinated / total) * 100).toFixed(1) : null
  const abstentionRate    = total > 0 ? ((abstained    / total) * 100).toFixed(1) : null

  return (
    <div style={styles.window}>
      {total > 0 && (
        <div style={styles.statsBar}>
          <StatChip label="Queries" value={total} color="var(--text-muted)" />
          <StatChip
            label="Hallucination Rate"
            value={`${hallucinationRate}%`}
            color={parseFloat(hallucinationRate) > 50 ? '#ef4444' : parseFloat(hallucinationRate) > 25 ? '#f59e0b' : '#22c55e'}
          />
          <StatChip
            label="Abstention Rate"
            value={`${abstentionRate}%`}
            color="#94a3b8"
          />
          <StatChip
            label="Answered"
            value={`${total - abstained}/${total}`}
            color="var(--accent-light)"
          />
        </div>
      )}

      <div style={styles.inner}>
        {messages.map(msg => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {loading && <TypingIndicator />}

        <div ref={bottomRef} style={{ height: '1px' }} />
      </div>
    </div>
  )
}

function StatChip({ label, value, color }) {
  return (
    <div style={styles.statChip}>
      <span style={styles.statLabel}>{label}</span>
      <span style={{ ...styles.statValue, color }}>{value}</span>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div style={styles.typingRow}>
      <div style={styles.avatar}>⬡</div>
      <div style={styles.typingBubble}>
        <span style={styles.dot} />
        <span style={{ ...styles.dot, animationDelay: '0.2s' }} />
        <span style={{ ...styles.dot, animationDelay: '0.4s' }} />
        <style>{`
          @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
            40%            { transform: translateY(-5px); opacity: 1; }
          }
        `}</style>
      </div>
    </div>
  )
}

const styles = {
  window: {
    flex:          1,
    overflowY:     'auto',
    display:       'flex',
    flexDirection: 'column',
  },
  statsBar: {
    display:        'flex',
    gap:            '8px',
    padding:        '10px 24px',
    borderBottom:   '1px solid var(--border)',
    background:     'var(--bg-sidebar)',
    flexWrap:       'wrap',
    flexShrink:     0,
  },
  statChip: {
    display:       'flex',
    flexDirection: 'column',
    alignItems:    'center',
    padding:       '4px 14px',
    borderRadius:  'var(--radius-sm)',
    border:        '1px solid var(--border)',
    background:    'var(--bg-primary)',
    minWidth:      '90px',
  },
  statLabel: {
    fontSize:      '10px',
    fontWeight:    600,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    color:         'var(--text-muted)',
  },
  statValue: {
    fontSize:   '16px',
    fontWeight: 700,
    marginTop:  '2px',
  },
  inner: {
    padding: '24px 0',
    maxWidth: '780px',
    margin:   '0 auto',
    padding:  '0 24px',
    display:  'flex',
    flexDirection: 'column',
    gap:      '24px',
  },
  typingRow: {
    display:    'flex',
    alignItems: 'flex-start',
    gap:        '12px',
  },
  avatar: {
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
  typingBubble: {
    display:      'flex',
    alignItems:   'center',
    gap:          '5px',
    background:   'var(--bg-message-ai)',
    border:       '1px solid var(--border)',
    borderRadius: '0 var(--radius) var(--radius) var(--radius)',
    padding:      '14px 18px',
  },
  dot: {
    display:         'inline-block',
    width:           '7px',
    height:          '7px',
    borderRadius:    '50%',
    background:      'var(--text-muted)',
    animation:       'bounce 1.2s infinite ease-in-out',
  },
}
