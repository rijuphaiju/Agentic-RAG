import ChatMessage from './ChatMessage'

export default function ChatWindow({ messages, loading, bottomRef }) {
  return (
    <div style={styles.window}>
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
    flex:      1,
    overflowY: 'auto',
    padding:   '24px 0',
  },
  inner: {
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
