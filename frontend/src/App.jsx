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
import { CAMERA_LOCATIONS, getCameraById } from './cameraLocations.js'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'
const MAX_HISTORY = 20
const MAX_ALERTS = 15
const ALERT_TTL_MS = 8000
const BG_SIM_MS = 10000

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
  const [error, setError] = useState('')
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

  // Load persisted history + stats once on page load so they survive refresh.
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [hRes, sRes] = await Promise.all([
          axios.get(`${API_URL}/history`),
          axios.get(`${API_URL}/stats`),
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
        const res = await axios.post(`${API_URL}/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 90000,
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
    tick()
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
    setPrediction(null)
    setError('')
  }

  const handleFileSelect = async (file) => {
    if (!file) return
    setLoading(true)
    setWakingUp(false)
    setError('')
    setPrediction(null)
    // After 5s switch the loading text to the cold-start message.
    const wakeTimer = setTimeout(() => setWakingUp(true), 5000)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('location', 'Uploaded Image')
      const postPredict = () =>
        axios.post(`${API_URL}/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 90000,
        })
      let res
      try {
        res = await postPredict()
      } catch (firstErr) {
        // Retry once after a 5s delay before surfacing the error.
        await new Promise((r) => setTimeout(r, 5000))
        res = await postPredict()
      }
      setPrediction(res.data)
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
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col transition-all duration-300">
      <AlertBanner alerts={activeAlerts} onDismiss={dismissAlert} />
      <header className="px-4 pt-10 sm:pt-14 pb-4 text-center animate-fade-up">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-[11px] sm:text-xs font-semibold text-slate-300 tracking-wide">
          <span className="relative flex h-2 w-2 text-emerald-400">
            <span className="status-ping absolute inline-flex h-full w-full" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
          </span>
          WORLDWIDE PROTECTION • 24 CAMERAS • NASA FIRMS LIVE
        </div>
        <h1 className="mt-3 text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight">
          <span className="animate-gradient-text">🔥 Forest Fire Detection System</span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base md:text-lg mt-3">
          {liveMode
            ? 'Live monitoring station — auto-scanning camera feeds'
            : 'Upload a satellite or forest image to detect fire risk'}
        </p>
        <div className="mt-4 flex justify-center">
          {!liveMode ? (
            <button
              type="button"
              onClick={enterLiveMode}
              className="btn-lift px-5 py-2.5 rounded-xl font-bold text-sm sm:text-base text-white bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 shadow-lg shadow-red-900/40 transition-all duration-300 active:scale-[0.99]"
            >
              📡 Switch to Live Monitoring
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setLiveMode(false)}
              className="btn-lift px-5 py-2.5 rounded-xl font-bold text-sm sm:text-base text-slate-100 bg-slate-700 hover:bg-slate-600 border border-slate-600 shadow-lg transition-all duration-300 active:scale-[0.99]"
            >
              ⏹ Stop Monitoring
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 w-full max-w-2xl mx-auto px-4 sm:px-6 pb-10 space-y-5 stagger">
        <div className="pt-2">
          <ModelInfoPanel apiUrl={API_URL} />
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
                <p className="text-center text-slate-400 text-sm">{wakingUp ? 'Waking up the server, this may take up to a minute on first request...' : 'Analyzing image...'}</p>
              </div>
            )}
            {error && (
              <div className="bg-red-950/60 border border-red-500/40 text-red-200 px-4 py-3 rounded-xl text-sm sm:text-base transition-all duration-300">
                {error}
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

      <footer className="py-6 text-center text-xs sm:text-sm text-slate-500">
        Built with TensorFlow, Flask &amp; React
      </footer>
    </div>
  )
}
