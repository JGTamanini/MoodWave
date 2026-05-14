import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  motion, AnimatePresence,
  useMotionValue, useTransform, animate
} from 'framer-motion'
import './App.css'

const API_URL = 'http://localhost:8000'

const MOOD_CONFIG = {
  'Melancólico': { emoji: '😢', accent: '#4a7fa5', glow: 'rgba(74,127,165,0.35)', orb1: '#1e3a5f', orb2: '#0a1f3d' },
  'Calmo':       { emoji: '😌', accent: '#3aafa9', glow: 'rgba(58,175,169,0.35)', orb1: '#1a4f4d', orb2: '#0a2e2d' },
  'Neutro':      { emoji: '😐', accent: '#8a8fa8', glow: 'rgba(138,143,168,0.3)', orb1: '#3a3d4a', orb2: '#1e2028' },
  'Animado':     { emoji: '😄', accent: '#e07b39', glow: 'rgba(224,123,57,0.4)',  orb1: '#5c2d0f', orb2: '#3a1800' },
  'Eufórico':    { emoji: '🤩', accent: '#f0c040', glow: 'rgba(240,192,64,0.45)', orb1: '#5a4200', orb2: '#3a2900' },
}

const DEFAULT_ACCENT = '#e07b39'
const DEFAULT_GLOW   = 'rgba(224,123,57,0.25)'

/* ── animated number counter ── */
function AnimatedNumber({ value, decimals = 0, suffix = '' }) {
  const mv = useMotionValue(0)
  const [display, setDisplay] = useState('0')

  useEffect(() => {
    const ctrl = animate(mv, value, { duration: 1.2, ease: 'easeOut' })
    const unsub = mv.on('change', v => setDisplay(v.toFixed(decimals) + suffix))
    return () => { ctrl.stop(); unsub() }
  }, [value])

  return <span>{display}</span>
}

/* ── wave svg (animated path) ── */
function AnimatedWave() {
  return (
    <motion.svg
      viewBox="0 0 200 36" fill="none" className="wave-svg"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.8, delay: 0.2 }}
    >
      <motion.path
        d="M0 18 Q16 4 32 18 Q48 32 64 18 Q80 4 96 18 Q112 32 128 18 Q144 4 160 18 Q176 32 192 18 Q196 14 200 18"
        stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" fill="none"
        animate={{ pathLength: [0.3, 1, 0.3] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
      />
    </motion.svg>
  )
}

/* ── animated probability bar ── */
function ProbBar({ label, value, color, delay = 0 }) {
  return (
    <motion.div className="prob-row"
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay }}
    >
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <motion.div
          className="prob-fill"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${(value * 100).toFixed(1)}%` }}
          transition={{ duration: 1, delay: delay + 0.2, ease: 'easeOut' }}
        />
      </div>
      <span className="prob-pct">{(value * 100).toFixed(1)}%</span>
    </motion.div>
  )
}

/* ── track item ── */
function TrackItem({ faixa, index, delay }) {
  return (
    <motion.li className="track-item"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ backgroundColor: 'rgba(255,255,255,0.05)', x: 4 }}
    >
      <span className="track-num">{String(index).padStart(2, '0')}</span>
      <div className="track-details">
        <span className="track-name">{faixa.titulo}</span>
        <span className="track-artist">{faixa.artista}</span>
      </div>
      <div className="track-stats">
        <span className="track-bpm">{Math.round(faixa.bpm)}<small>bpm</small></span>
        <div className="ene-bar">
          <motion.div
            className="ene-fill"
            initial={{ width: 0 }}
            animate={{ width: `${(faixa.energia * 100).toFixed(0)}%` }}
            transition={{ duration: 0.8, delay: delay + 0.2, ease: 'easeOut' }}
          />
        </div>
      </div>
    </motion.li>
  )
}

/* ── floating orb background ── */
function OrbBackground({ mood }) {
  const cfg = mood ? MOOD_CONFIG[mood] : null
  return (
    <div className="orb-layer" aria-hidden>
      <motion.div className="orb orb-1"
        animate={{ background: cfg ? cfg.orb1 : '#3a1800' }}
        transition={{ duration: 1.4, ease: 'easeInOut' }}
      />
      <motion.div className="orb orb-2"
        animate={{ background: cfg ? cfg.orb2 : '#1a0f00' }}
        transition={{ duration: 1.4, ease: 'easeInOut' }}
      />
      <div className="orb orb-3" />
    </div>
  )
}

/* ══════════════════════════════════════════ */
export default function App() {
  const [texto,   setTexto]   = useState('')
  const [energia, setEnergia] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [result,  setResult]  = useState(null)

  const mood    = result ? (MOOD_CONFIG[result.humor] ?? null) : null
  const accent  = mood?.accent  ?? DEFAULT_ACCENT
  const glow    = mood?.glow    ?? DEFAULT_GLOW
  const energyPct = Math.round(energia * 100)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!texto.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await axios.post(`${API_URL}/analisar`, {
        texto: texto.trim(), energia,
      })
      setResult(data)
    } catch (err) {
      setError(err.code === 'ERR_NETWORK'
        ? 'Não foi possível conectar à API. Verifique se o backend está rodando em localhost:8000.'
        : (err.response?.data?.detail ?? 'Ocorreu um erro inesperado.'))
    } finally {
      setLoading(false)
    }
  }

  /* stagger for header children */
  const headerStagger = {
    hidden: {},
    visible: { transition: { staggerChildren: 0.12 } },
  }
  const fadeUp = {
    hidden:  { opacity: 0, y: 24 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.7, ease: 'easeOut' } },
  }

  return (
    <div className="app">
      <div className="grain" />
      <OrbBackground mood={result?.humor} />

      {/* ── HEADER ── */}
      <motion.header className="header"
        variants={headerStagger} initial="hidden" animate="visible"
      >
        <motion.div className="logo-wrap" variants={fadeUp}>
          <motion.div className="logo-icon"
            style={{ color: accent }}
            animate={{ color: accent }}
            transition={{ duration: 0.8 }}
          >
            <AnimatedWave />
          </motion.div>
          <h1 className="logo-name">MoodWave</h1>
        </motion.div>
        <motion.p className="tagline" variants={fadeUp}>
          Descreva seu humor · Receba sua trilha sonora
        </motion.p>
      </motion.header>

      <main className="main">

        {/* ── INPUT CARD ── */}
        <motion.form className="card input-card" onSubmit={handleSubmit}
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <div className="field">
            <label className="field-label" htmlFor="mood-input">
              Como você está se sentindo?
            </label>
            <motion.textarea
              id="mood-input"
              className="mood-textarea"
              placeholder="Ex: Hoje foi um dia incrível, me sinto leve e animado com tudo que aconteceu…"
              value={texto}
              onChange={e => setTexto(e.target.value)}
              rows={4}
              disabled={loading}
              whileFocus={{ borderColor: accent, boxShadow: `0 0 0 3px ${glow}` }}
              transition={{ duration: 0.25 }}
            />
          </div>

          <div className="field">
            <div className="slider-header">
              <label className="field-label" htmlFor="energy-range">Nível de Energia</label>
              <motion.span className="energy-badge"
                animate={{ color: accent, borderColor: `${accent}55` }}
                transition={{ duration: 0.6 }}
              >
                {energyPct}%
              </motion.span>
            </div>
            <div className="slider-row">
              <span className="slider-icon">😴</span>
              <input
                id="energy-range" type="range" className="energy-slider"
                min="0" max="1" step="0.01"
                value={energia}
                onChange={e => setEnergia(parseFloat(e.target.value))}
                disabled={loading}
                style={{
                  '--pct': `${energyPct}%`,
                  '--accent': accent,
                }}
              />
              <span className="slider-icon">⚡</span>
            </div>
          </div>

          <motion.button
            className="submit-btn" type="submit"
            disabled={loading || !texto.trim()}
            animate={{ background: accent, boxShadow: `0 4px 32px ${glow}` }}
            transition={{ duration: 0.6 }}
            whileHover={!loading && texto.trim() ? { scale: 1.02, boxShadow: `0 6px 40px ${glow}` } : {}}
            whileTap={!loading && texto.trim()  ? { scale: 0.97 } : {}}
          >
            <AnimatePresence mode="wait">
              {loading ? (
                <motion.span key="loading" className="dots"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                >
                  <span /><span /><span />
                </motion.span>
              ) : (
                <motion.span key="label"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                >
                  Analisar Humor
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>
        </motion.form>

        {/* ── ERROR ── */}
        <AnimatePresence>
          {error && (
            <motion.div className="card error-card"
              initial={{ opacity: 0, y: -10, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.35 }}
            >
              <span className="error-icon">⚠</span>
              <p>{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── RESULT ── */}
        <AnimatePresence>
          {result && mood && (
            <motion.div className="card result-card"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
            >

              {/* mood hero */}
              <motion.div className="mood-hero"
                initial={{ scale: 0.88, opacity: 0 }}
                animate={{
                  scale: 1, opacity: 1,
                  background: `radial-gradient(ellipse at 30% 50%, ${mood.orb1}cc, ${mood.orb2}99)`,
                  boxShadow: `0 0 60px ${mood.glow}, inset 0 1px 0 rgba(255,255,255,0.08)`,
                }}
                transition={{ duration: 0.8, ease: [0.34, 1.56, 0.64, 1] }}
              >
                <motion.span className="mood-emoji"
                  initial={{ scale: 0, rotate: -20 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ duration: 0.7, delay: 0.2, ease: [0.34, 1.56, 0.64, 1] }}
                >
                  {mood.emoji}
                </motion.span>
                <div className="mood-meta">
                  <motion.span className="mood-label"
                    initial={{ opacity: 0, x: 16 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                  >
                    {result.humor}
                  </motion.span>
                  <motion.span className="mood-sub"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.5 }}
                  >
                    score&nbsp;
                    <AnimatedNumber value={result.score_humor * 100} decimals={0} suffix="%" />
                  </motion.span>
                </div>

                {/* pulsing glow ring */}
                <motion.div className="glow-ring"
                  style={{ borderColor: mood.accent }}
                  animate={{ scale: [1, 1.06, 1], opacity: [0.4, 0.15, 0.4] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                />
              </motion.div>

              {/* score track */}
              <motion.div className="score-track-wrap"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <div className="score-track">
                  <motion.div className="score-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${(result.score_humor * 100).toFixed(1)}%` }}
                    transition={{ duration: 1.2, delay: 0.5, ease: 'easeOut' }}
                  />
                  <motion.div className="score-thumb"
                    style={{ borderColor: accent, boxShadow: `0 0 12px ${glow}` }}
                    initial={{ left: 0 }}
                    animate={{ left: `${(result.score_humor * 100).toFixed(1)}%` }}
                    transition={{ duration: 1.2, delay: 0.5, ease: 'easeOut' }}
                  />
                </div>
                <div className="score-axis">
                  <span>😢</span><span>😌</span><span>😐</span><span>😄</span><span>🤩</span>
                </div>
              </motion.div>

              {/* prob bars */}
              <div className="probs-section">
                <motion.h3 className="section-title"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 }}
                >
                  Sentimento detectado
                </motion.h3>
                <ProbBar label="Positivo" value={result.probabilidades.positivo} color="#4caf82" delay={0.65} />
                <ProbBar label="Neutro"   value={result.probabilidades.neutro}   color="#8a8fa8" delay={0.78} />
                <ProbBar label="Negativo" value={result.probabilidades.negativo} color="#c0504a" delay={0.91} />
              </div>

              {/* stats strip */}
              <motion.div className="stats-strip"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.5 }}
              >
                {[
                  { val: result.estatisticas.energia_media * 100,  dec: 0, suf: '%',   lbl: 'Energia' },
                  { val: result.estatisticas.valencia_media * 100, dec: 0, suf: '%',   lbl: 'Valência' },
                  { val: result.estatisticas.bpm_medio,            dec: 0, suf: ' bpm', lbl: 'BPM médio' },
                ].map((s, i) => (
                  <div key={i} className="stat">
                    <span className="stat-val" style={{ color: accent }}>
                      <AnimatedNumber value={s.val} decimals={s.dec} suffix={s.suf} />
                    </span>
                    <span className="stat-lbl">{s.lbl}</span>
                  </div>
                ))}
              </motion.div>

              {/* playlist */}
              <div className="playlist-section">
                <motion.h3 className="section-title"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  transition={{ delay: 1.1 }}
                >
                  Sua Playlist
                </motion.h3>
                <ol className="track-list">
                  {result.playlist.map((faixa, i) => (
                    <TrackItem
                      key={i} index={i + 1} faixa={faixa}
                      delay={1.15 + i * 0.06}
                    />
                  ))}
                </ol>
              </div>

            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <motion.footer className="footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        PLN · Fuzzy Sugeno · Algoritmo Genético — IA · Católica SC
      </motion.footer>
    </div>
  )
}
