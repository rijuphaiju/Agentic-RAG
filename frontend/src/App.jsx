import { useState, useRef, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'
import EvalTable from './components/EvalTable'

const WELCOME = {
  id: 'welcome',
  role: 'assistant',
  text: "Hello! I'm HARA — the Hallucination-Aware Retrieval Agent. Ask me anything — I'll retrieve relevant passages and give you a verified, grounded answer.",
  metadata: null,
  stage: null,
}

export default function App() {
  const [view, setView]             = useState('chat')
  const [stage, setStage]           = useState('stage4')
  const [messages, setMessages]     = useState([WELCOME])
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [sessionStats, setSession]  = useState({ total: 0, hallucinated: 0, abstained: 0 })
  const bottomRef                   = useRef(null)

  useEffect(() => {
    if (view === 'chat') {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, loading, view])

  async function sendMessage(question) {
    if (!question.trim() || loading) return
    setError(null)

    const userMsg = {
      id:       Date.now(),
      role:     'user',
      text:     question.trim(),
      metadata: null,
      stage:    null,
    }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: question.trim(), stage }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Request failed')
      }

      const data    = await res.json()
      const meta    = data.metadata || {}
      const scores  = meta.scores || meta.verification_scores || {}
      const hasVerifier  = data.stage !== 'stage1'   // Stage 1 has no verifier
      const hallProb     = (scores.PARTIAL || 0) + (scores.UNSUPPORTED || 0)
      const abstained    = meta.status === 'ABSTAINED'
      const hallucinated = hasVerifier && !abstained && hallProb >= 0.5

      setSession(prev => ({
        total:       prev.total + 1,
        hallucinated: prev.hallucinated + (hallucinated ? 1 : 0),
        abstained:   prev.abstained + (abstained ? 1 : 0),
      }))

      const aiMsg = {
        id:       Date.now() + 1,
        role:     'assistant',
        text:     data.answer,
        metadata: data.metadata,
        stage:    data.stage,
      }
      setMessages(prev => [...prev, aiMsg])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function clearChat() {
    setMessages([WELCOME])
    setError(null)
    setSession({ total: 0, hallucinated: 0, abstained: 0 })
  }

  return (
    <div style={styles.layout}>
      <Sidebar
        view={view}
        onViewChange={setView}
        stage={stage}
        onStageChange={setStage}
        onClear={clearChat}
      />

      <div style={styles.main}>
        {view === 'chat' ? (
          <>
            <ChatWindow messages={messages} loading={loading} bottomRef={bottomRef} sessionStats={sessionStats} />

            {error && (
              <div style={styles.errorBanner}>
                <span>⚠</span> {error}
              </div>
            )}

            <ChatInput onSend={sendMessage} disabled={loading} />
          </>
        ) : (
          <EvalTable />
        )}
      </div>
    </div>
  )
}

const styles = {
  layout: {
    display:  'flex',
    height:   '100vh',
    overflow: 'hidden',
  },
  main: {
    flex:          1,
    display:       'flex',
    flexDirection: 'column',
    overflow:      'hidden',
  },
  errorBanner: {
    display:      'flex',
    alignItems:   'center',
    gap:          '8px',
    margin:       '0 24px 4px',
    padding:      '10px 16px',
    background:   'rgba(239,68,68,0.1)',
    border:       '1px solid rgba(239,68,68,0.3)',
    borderRadius: 'var(--radius-sm)',
    color:        '#fca5a5',
    fontSize:     '13px',
  },
}
