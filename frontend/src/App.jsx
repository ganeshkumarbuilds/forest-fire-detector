import { useCallback, useEffect, useState } from 'react'
import axios from 'axios'
import UploadPanel from './components/UploadPanel.jsx'
import ResultCard, { getRiskLevel } from './components/ResultCard.jsx'
import HistoryPanel from './components/HistoryPanel.jsx'
import LiveMonitor, { SAMPLE_IMAGES as BG_SAMPLE_IMAGES } from './components/LiveMonitor.jsx'
import MapView from './components/MapView.jsx'
import AlertBanner from './components/AlertBanner.jsx'
import AlertLog from './components/AlertLog.jsx'
import ModelInfoPanel from './components/ModelInfoPanel.jsx'
import ErrorAnalysisPanel from './components/ErrorAnalysisPanel.jsx'
import RobustnessPanel from './components/RobustnessPanel.jsx'
import { CAMERA_LOCATIONS, getCameraById } from './cameraLocations.js'
import waitForBackend from './waitForBackend.js'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'
const MAX_HISTORY = 20
const MAX_ALERTS = 15
const ALERT_TTL_MS = 8000
// Render free tier: cold start 50s+ + TF load. 3min timeout + retries.
// Background sim throttled to 60s so free instance is never hammered.
const PREDICT_TIMEOUT_MS = 180000
const BG_SIM_MS = 300000

function revokeIfBlob(url) {
  if (typeof url === 'string' && url.startsWith('blob:')) {
    try {
      URL.revokeObjectURL(url)
    } catch {
      // ignore — URL may already be revoked
    }
  }
}

export default function App() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  // True while a /predict request has been in flight for >5s (Render cold start).
  const [wakingUp, setWakingUp] = useState(false)
  // Seconds elapsed while waiting for the sleeping backend to warm up.
  const [warmSecs, setWarmSecs] = useState(0)
  const [error, setError] = useState('')
  const [errorQuality, setErrorQuality] = useState(null)
  const [history, setHistory] = useState([])
  const [liveMode, setLiveMode] = useState(false)
  // All-time server stats (GET /stats). Null until loaded or if backend
  // is unreachable — HistoryPanel then falls back to local-history stats.
  const [serverStats, setServerStats] = useState(null)
  // Latest detection per camera for the map. Fed by the background
  // simulation (map-only) and by manual live-monitoring results.
  // Manual uploads carry no camera, so they never touch the map.
  const [cameraStatus, setCameraStatus] = useState({})
  // Critical-risk alerts: active toasts + permanent session log.
  const [activeAlerts, setActiveAlerts] = useState([])
  const [alertLog, setAlertLog] = useState([])

  const dismissAlert = useCallback((id) => {
    setActiveAlerts((prev) => prev.filter((a) => a.id !== id))
  }, [])

  // Fires ONLY for CRITICAL RISK detections — never for high/low/safe.
  const triggerAlert = useCallback(
    ({ location, fire_detected, confidence, timestamp }) => {
      if (getRiskLevel(fire_detected, confidence).key !== 'critical') return
      const alert = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        location,
        riskLabel: 'CRITICAL RISK',
        confidence: confidence ?? 0,
        timestamp,
      }
      setActiveAlerts((prev) => [alert, ...prev])
      setAlertLog((prev) => [alert, ...prev].slice(0, MAX_ALERTS))
      setTimeout(() => dismissAlert(alert.id), ALERT_TTL_MS)
    },
    [dismissAlert]
  )

  // Keep Render free instance warm:
  // 1) Aggressive warm-up on page load (retry /health for ~2 min)
  // 2) Heartbeat every 4 min while tab is open → prevents 15-min sleep
  // This makes "Waking up..." happen ONCE, then instant.
  useEffect(() => {
    let cancelled = false
    const ping = async () => {
      try {
        await axios.get(`${API_URL}/health`, { timeout: 15000 })
        return true
      } catch {
        return false
      }
    }
    const warm = async () => {
      for (let i = 0; i < 12 && !cancelled; i++) {
        if (await ping()) return
        await new Promise((r) => setTimeout(r, 10000))
      }
    }
    warm()
    const hb = setInterval(() => {
      if (!document.hidden) ping()
    }, 4 * 60 * 1000)
    const onVisible = () => {
      if (!document.hidden) ping()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      cancelled = true
      clearInterval(hb)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])

  // Load persisted history + stats once on page load so they survive refresh.
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [hRes, sRes] = await Promise.all([
          axios.get(`${API_URL}/history`, { timeout: 60000 }),
          axios.get(`${API_URL}/stats`, { timeout: 60000 }),
        ])
        if (cancelled) return
        const items = Array.isArray(hRes.data) ? hRes.data : []
        setHistory(
          items.slice(0, MAX_HISTORY).map((e, i) => ({
            id: `server-${e.timestamp ?? i}-${i}`,
            imageUrl: null, // server stores predictions, not images
            fileName: e.filename ?? 'image',
            fire_detected: !!e.fire_detected,
            confidence: e.confidence ?? 0,
            timestamp: e.timestamp,
          }))
        )
        // Seed map markers from persisted filenames ("Camera 0X — ..."),
        // most recent entry per camera wins.
        const seeded = {}
        for (const e of items) {
          const m = /Camera 0?(\d{1,2})/i.exec(e.filename ?? '')
          if (!m) continue
          const n = parseInt(m[1], 10)
          if (Number.isNaN(n) || n < 1 || n > CAMERA_LOCATIONS.length) continue
          const id = `camera-${String(n).padStart(2, '0')}`
          if (!seeded[id]) {
            seeded[id] = {
              fire_detected: !!e.fire_detected,
              confidence: e.confidence ?? 0,
              timestamp: e.timestamp,
            }
          }
        }
        setCameraStatus(seeded)
        setServerStats(sRes.data)
      } catch {
        // Backend without these endpoints or unreachable — stay session-only.
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  // Background map simulation (auto-starts on page load, no click needed):
  // every BG_SIM_MS analyzes ONE camera (rotating through all of them) with
  // a randomly selected sample image, updating only marker colors. It never
  // touches history, stats, or alerts — manual Live Monitoring is unchanged.
  useEffect(() => {
    let cancelled = false
    let busy = false
    let idx = 0
    const tick = async () => {
      if (cancelled || busy || CAMERA_LOCATIONS.length === 0) return
      busy = true
      try {
        const cam = CAMERA_LOCATIONS[idx % CAMERA_LOCATIONS.length]
        idx += 1
        const sample =
          BG_SAMPLE_IMAGES[Math.floor(Math.random() * BG_SAMPLE_IMAGES.length)]
        const imgRes = await fetch(sample.url)
        if (!imgRes.ok) return
        const blob = await imgRes.blob()
        const file = new File([blob], sample.url.split('/').pop() || 'sample.jpg', {
          type: blob.type || 'image/jpeg',
        })
        const formData = new FormData()
        formData.append('file', file)
        formData.append('location', `${cam.name} (${cam.zone})`)
        // Map-only tick: skip heatmap (Grad-CAM costs a 2nd inference
        // pass; the map only uses fire/confidence/timestamp). Manual
        // uploads and LiveMonitor still request heatmap=1 unchanged.
        const res = await axios.post(`${API_URL}/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: PREDICT_TIMEOUT_MS,
        })
        if (cancelled) return
        setCameraStatus((prev) => ({
          ...prev,
          [cam.id]: {
            fire_detected: res.data.fire_detected,
            confidence: res.data.confidence,
            timestamp: res.data.timestamp,
          },
        }))
      } catch {
        // Silent — background liveliness must never surface errors.
      } finally {
        busy = false
      }
    }
    // Wait for a cold backend to finish warming up BEFORE the first
    // background /predict — firing immediately on page load hammers a
    // still-loading free-tier instance (503s + long waits). Interval,
    // rotation, and map-only updates are unchanged.
    waitForBackend(API_URL, { timeoutMs: 240000 }).then((ok) => {
      if (cancelled || !ok) return
      tick()
    })
    const id = setInterval(tick, BG_SIM_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  // Optimistically bump all-time stats when a new detection lands,
  // so numbers stay correct without needing a refetch.
  const bumpStats = useCallback((fire_detected, confidence) => {
    setServerStats((prev) => {
      if (!prev) return prev
      const prevTotal = prev.total_analyzed ?? 0
      const total = prevTotal + 1
      const prevAvg = prev.average_confidence ?? 0
      return {
        total_analyzed: total,
        fire_count: (prev.fire_count ?? 0) + (fire_detected ? 1 : 0),
        safe_count: (prev.safe_count ?? 0) + (fire_detected ? 0 : 1),
        average_confidence:
          (prevAvg * prevTotal + (confidence ?? 0)) / total,
      }
    })
  }, [])

  const addToHistory = useCallback((entry) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const newEntry = { id, ...entry }
    setHistory((prev) => {
      const next = [newEntry, ...prev].slice(0, MAX_HISTORY)
      // Revoke blob thumbnails that fell off the list to avoid leaks
      const evicted = [newEntry, ...prev].slice(MAX_HISTORY)
      evicted.forEach((e) => revokeIfBlob(e.imageUrl))
      return next
    })
  }, [])

  const clearHistory = useCallback(() => {
    setHistory((prev) => {
      prev.forEach((e) => revokeIfBlob(e.imageUrl))
      return []
    })
  }, [])

  const handleReset = () => {
    if (prediction?.original_image) URL.revokeObjectURL(prediction.original_image)
    setPrediction(null)
    setError('')
    setErrorQuality(null)
  }

  const handleFileSelect = async (file) => {
    if (!file) return
    setLoading(true)
    setWakingUp(false)
    setWarmSecs(0)
    setError('')
    setErrorQuality(null)
    setPrediction(null)
    // After 5s switch the loading text to the cold-start message.
    const wakeTimer = setTimeout(() => setWakingUp(true), 5000)
    try {
      // Cold-boot gate: free hosting sleeps when idle and its proxy kills
      // requests hanging >~100s. So wait for /ready with cheap fast polls
      // FIRST (with a live seconds counter), then send /predict exactly
      // once — it can no longer hang through a cold start.
      setWakingUp(true)
      const ready = await waitForBackend(API_URL, {
        timeoutMs: 240000,
        onTick: (s) => setWarmSecs(s),
      })
      if (!ready) {
        throw new Error(
          'Server is still starting (free hosting sleeps when idle). Please wait about a minute and press Analyze Image again.'
        )
      }
      const formData = new FormData()
      formData.append('file', file)
      formData.append('location', 'Uploaded Image')
      const postPredict = () =>
        axios.post(`${API_URL}/predict?heatmap=1`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: PREDICT_TIMEOUT_MS,
        })
      let res
      try {
        res = await postPredict()
      } catch (e) {
        // 503 + retry flag = model still warming: brief re-wait, one retry.
        if (e.response?.status !== 503 || !e.response?.data?.retry) throw e
        const readyAgain = await waitForBackend(API_URL, {
          timeoutMs: 120000,
          onTick: (s) => setWarmSecs(s),
        })
        if (!readyAgain) throw e
        res = await postPredict()
      }
      setPrediction({ ...res.data, original_image: URL.createObjectURL(file) })
      addToHistory({
        imageUrl: URL.createObjectURL(file),
        fileName: file.name,
        fire_detected: res.data.fire_detected,
        confidence: res.data.confidence,
        timestamp: res.data.timestamp,
      })
      bumpStats(res.data.fire_detected, res.data.confidence)
      triggerAlert({
        location: 'Uploaded Image',
        fire_detected: res.data.fire_detected,
        confidence: res.data.confidence,
        timestamp: res.data.timestamp,
      })
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Prediction failed')
      if (err.response?.data?.quality) {
        setErrorQuality(err.response.data.quality)
      }
    } finally {
      clearTimeout(wakeTimer)
      setWakingUp(false)
      setLoading(false)
    }
  }

  const handleLiveResult = useCallback(
    (payload) => {
      addToHistory(payload)
      bumpStats(payload.fire_detected, payload.confidence)
      // Map reflects live cameras only; manual uploads have no cameraId.
      if (payload.cameraId) {
        setCameraStatus((prev) => ({
          ...prev,
          [payload.cameraId]: {
            fire_detected: payload.fire_detected,
            confidence: payload.confidence,
            timestamp: payload.timestamp,
          },
        }))
      }
      const cam = payload.cameraId ? getCameraById(payload.cameraId) : null
      triggerAlert({
        location: cam ? `${cam.name} (${cam.zone})` : payload.fileName || 'Live Camera',
        fire_detected: payload.fire_detected,
        confidence: payload.confidence,
        timestamp: payload.timestamp,
      })
    },
    [addToHistory, bumpStats, triggerAlert]
  )

  const enterLiveMode = () => {
    setPrediction(null)
    setError('')
    setLiveMode(true)
  }

  return (
    <div className="min-h-screen bg-[#0b1220] text-slate-100 flex flex-col selection:bg-orange-500/30">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(249,115,22,0.08),transparent_60%),radial-gradient(40%_30%_at_90%_10%,rgba(34,211,238,0.06),transparent_60%)]" />
      <AlertBanner alerts={activeAlerts} onDismiss={dismissAlert} />
      <nav className="sticky top-0 z-40 backdrop-blur-xl bg-slate-900/65 border-b border-slate-800/80">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-[56px] flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <span className="h-9 w-9 shrink-0 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 grid place-items-center shadow-lg shadow-orange-900/20 text-[16px]">🔥</span>
            <div className="leading-tight">
              <p className="font-bold tracking-tight text-[15px] sm:text-[16px]">Forest Fire Detection</p>
              <p className="hidden sm:block text-[11px] text-slate-400 -mt-0.5">AI monitoring • TFLite • FD-CAM</p>
            </div>
            <span className="hidden sm:inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" /> Live
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden md:inline-flex items-center gap-1.5 text-xs text-slate-400 border border-slate-700/60 rounded-full px-3 py-1.5 bg-slate-800/40">
              <span className="h-2 w-2 rounded-full bg-emerald-400" /> NASA FIRMS • 24 cameras
            </span>
            <span className="text-xs text-slate-500 hidden sm:inline">v1.0</span>
          </div>
        </div>
      </nav>

      <header className="max-w-5xl mx-auto w-full px-4 sm:px-6 pt-8 sm:pt-10 pb-6 text-center">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/60 text-[11px] sm:text-xs font-semibold tracking-widest text-slate-300 backdrop-blur">
          <span className="relative flex h-2 w-2 text-emerald-400">
            <span className="status-ping absolute inline-flex h-full w-full" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
          </span>
          WORLDWIDE PROTECTION • 24 CAMERAS • NASA FIRMS LIVE
        </div>
        <h1 className="mt-4 text-[28px] sm:text-[36px] md:text-[44px] font-extrabold tracking-[-0.02em] leading-[0.95]">
          <span className="bg-gradient-to-r from-orange-300 via-amber-300 to-orange-400 bg-clip-text text-transparent">Forest Fire</span>
          <span className="text-white"> Detection System</span>
        </h1>
        <p className="mx-auto max-w-2xl text-slate-400 text-[14px] sm:text-[15px] leading-relaxed mt-3">
          {liveMode
            ? 'Live monitoring station — auto-scanning camera feeds across 24 worldwide zones.'
            : 'Upload a satellite or forest image to detect fire risk — instant prediction with visual explainability.'}
        </p>
        <div className="mt-6 flex justify-center">
          {!liveMode ? (
            <button
              type="button"
              onClick={enterLiveMode}
              className="btn-lift inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm text-white bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 shadow-lg shadow-red-900/25"
            >
              <span className="text-base">📡</span> Switch to Live Monitoring
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setLiveMode(false)}
              className="btn-lift inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 shadow"
            >
              ⏹ Stop Monitoring
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 w-full max-w-3xl mx-auto px-4 sm:px-6 pb-10 space-y-6">
        <div className="pt-2">
          <ModelInfoPanel apiUrl={API_URL} />
        </div>
        <div className="pt-2">
          <ErrorAnalysisPanel apiUrl={API_URL} />
        </div>
        <div className="pt-2">
          <RobustnessPanel apiUrl={API_URL} />
        </div>

        {liveMode ? (
          <LiveMonitor apiUrl={API_URL} onLiveResult={handleLiveResult} />
        ) : (
          <>
            <UploadPanel
              onFileSelect={handleFileSelect}
              loading={loading}
              hasResult={!!prediction}
              onReset={handleReset}
            />

            {loading && (
              <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5 space-y-3 animate-pulse">
                <div className="h-4 bg-slate-700 rounded w-1/3" />
                <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full w-1/2 bg-gradient-to-r from-orange-500 to-red-500 rounded-full" />
                </div>
                <p className="text-center text-slate-400 text-sm">{wakingUp ? `Warming up the server… ${warmSecs}s (free hosting sleeps when idle, one moment)` : 'Analyzing image...'}</p>
              </div>
            )}
            {error && (
              <div className="bg-red-950/60 border border-red-500/40 text-red-200 px-4 py-3 rounded-xl text-sm sm:text-base transition-all duration-300">
                {error}
              </div>
            )}
            {errorQuality && (
              <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 p-5 sm:p-6 space-y-4 animate-fade-up transition-all duration-300">
                <div className="flex items-center justify-between">
                  <h3 className="text-red-400 text-sm sm:text-base font-semibold flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-red-400"></span>
                    Image Quality Check Failed
                  </h3>
                  <span className="text-red-300 text-xs px-2 py-1 rounded bg-red-950/50">
                    Rejected
                  </span>
                </div>
                
                {errorQuality.width && errorQuality.height && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center">
                    <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
                      <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Dimensions</p>
                      <p className="text-slate-100 font-mono text-sm sm:text-base">
                        {errorQuality.width} × {errorQuality.height}
                      </p>
                    </div>
                    {errorQuality.blur_score !== undefined && (
                      <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
                        <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Blur Score</p>
                        <p className="text-slate-100 font-mono text-sm sm:text-base">
                          {errorQuality.blur_score?.toFixed?.(1) ?? '—'}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-1">(higher = sharper)</p>
                      </div>
                    )}
                    {errorQuality.brightness !== undefined && (
                      <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
                        <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Brightness</p>
                        <p className="text-slate-100 font-mono text-sm sm:text-base">
                          {errorQuality.brightness?.toFixed?.(1) ?? '—'} / 255
                        </p>
                      </div>
                    )}
                    {errorQuality.contrast !== undefined && (
                      <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
                        <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Contrast</p>
                        <p className="text-slate-100 font-mono text-sm sm:text-base">
                          {errorQuality.contrast?.toFixed?.(1) ?? '—'}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {errorQuality.warnings && errorQuality.warnings.length > 0 && (
                  <div className="bg-amber-950/50 border border-amber-500/40 rounded-xl p-3">
                    <p className="text-amber-200 text-xs font-medium mb-1 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                      Quality Issues
                    </p>
                    <ul className="text-amber-100 text-xs space-y-1">
                      {errorQuality.warnings.map((w, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <span className="w-1 h-1 rounded-full bg-amber-400 mt-1.5 flex-shrink-0"></span>
                          <span>{w}</span>
                        </li>
                      ))}
                    </ul>
                    <p className="text-amber-500 text-[10px] mt-2">
                      These are heuristic indicators, not OOD detection.
                    </p>
                  </div>
                )}
                
                {errorQuality.reason && (
                  <p className="text-red-300 text-xs text-center">
                    Rejection reason: {errorQuality.reason.replace(/_/g, ' ')}
                  </p>
                )}
              </div>
            )}
            {prediction && !loading && (
              <div className="mt-2 pt-6 border-t border-slate-700/60">
                <ResultCard result={prediction} />
              </div>
            )}
          </>
        )}

        <div className="pt-2">
          <HistoryPanel history={history} onClear={clearHistory} stats={serverStats} />
        </div>

        <div className="pt-2">
          <MapView cameraStatus={cameraStatus} apiUrl={API_URL} />
        </div>

        <div className="pt-2">
          <AlertLog alerts={alertLog} apiUrl={API_URL} />
        </div>
      </main>

      <footer className="mt-2 border-t border-slate-800/80 py-6 text-center">
        <p className="text-xs tracking-wide text-slate-500">Built with TensorFlow • Flask • React • Tailwind • Leaflet</p>
        <p className="text-[11px] text-slate-600 mt-1">Offline evaluation artifacts • Deterministic 70/15/15 split • Threshold 0.5</p>
      </footer>
    </div>
  )
}
