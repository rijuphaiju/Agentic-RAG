import { useState, useRef, useEffect } from 'react'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    autoResize()
  }, [value])

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q)
    setValue('')
  }

  return (
    <div style={styles.wrapper}>
      <div style={styles.container}>
        <div style={{ ...styles.box, ...(disabled ? styles.boxDisabled : {}) }}>
          <textarea
            ref={textareaRef}
            style={styles.textarea}
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about history, science, or anything in HotpotQA..."
            disabled={disabled}
            rows={1}
          />
          <button
            style={{
              ...styles.sendBtn,
              ...((!value.trim() || disabled) ? styles.sendBtnDisabled : {}),
            }}
            onClick={submit}
            disabled={!value.trim() || disabled}
            title="Send (Enter)"
          >
            {disabled ? <Spinner /> : <SendIcon />}
          </button>
        </div>
        <div style={styles.hint}>
          Press <kbd style={styles.kbd}>Enter</kbd> to send · <kbd style={styles.kbd}>Shift+Enter</kbd> for newline
        </div>
      </div>
    </div>
  )
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}

function Spinner() {
  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
      <div style={{
        width:  '16px',
        height: '16px',
        border: '2px solid rgba(255,255,255,0.3)',
        borderTopColor: '#fff',
        borderRadius: '50%',
        animation: 'spin 0.7s linear infinite',
      }} />
    </>
  )
}

const styles = {
  wrapper: {
    padding:         '12px 24px 20px',
    background:      'linear-gradient(to top, var(--bg-primary) 80%, transparent)',
  },
  container: {
    maxWidth: '780px',
    margin:   '0 auto',
    display:  'flex',
    flexDirection:'column',
    gap:      '6px',
  },
  box: {
    display:      'flex',
    alignItems:   'flex-end',
    gap:          '10px',
    background:   'var(--bg-input)',
    border:       '1px solid var(--border-light)',
    borderRadius: 'var(--radius)',
    padding:      '12px 14px',
    transition:   'border-color 0.15s',
    boxShadow:    '0 0 0 0 transparent',
  },
  boxDisabled: {
    opacity: 0.6,
  },
  textarea: {
    flex:       1,
    background: 'transparent',
    border:     'none',
    outline:    'none',
    resize:     'none',
    color:      'var(--text-primary)',
    fontSize:   '15px',
    lineHeight: 1.6,
    fontFamily: 'var(--font)',
    minHeight:  '24px',
    maxHeight:  '200px',
    overflowY:  'auto',
  },
  sendBtn: {
    width:          '36px',
    height:         '36px',
    borderRadius:   'var(--radius-sm)',
    background:     'var(--accent)',
    border:         'none',
    color:          '#fff',
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'center',
    cursor:         'pointer',
    flexShrink:     0,
    transition:     'background 0.15s, transform 0.1s',
  },
  sendBtnDisabled: {
    background: 'var(--bg-active)',
    cursor:     'not-allowed',
    color:      'var(--text-muted)',
  },
  hint: {
    fontSize:       '12px',
    color:          'var(--text-muted)',
    textAlign:      'center',
  },
  kbd: {
    background:   'var(--bg-active)',
    border:       '1px solid var(--border-light)',
    borderRadius: '3px',
    padding:      '1px 5px',
    fontSize:     '11px',
    fontFamily:   'monospace',
  },
}
