import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import ResultCard, { getRiskLevel } from './ResultCard.jsx'
import { getCameraById } from '../cameraLocations.js'
import waitForBackend from '../waitForBackend.js'

// Place these 5 images in frontend/public/samples/ (see public/samples/README.md)
// `id` values match frontend/src/cameraLocations.js so map markers stay in sync.
export const SAMPLE_IMAGES = [
  { id: 'camera-01', url: '/samples/sample-1.jpg', label: 'Camera 01' },
  { id: 'camera-02', url: '/samples/sample-2.jpg', label: 'Camera 02' },
  { id: 'camera-03', url: '/samples/sample-3.jpg', label: 'Camera 03' },
  { id: 'camera-04', url: '/samples/sample-4.jpg', label: 'Camera 04' },
  { id: 'camera-05', url: '/samples/sample-5.jpg', label: 'Camera 05' },
]

const CYCLE_MS = 6000
const WEBCAM_MS = 6000

function fileNameFromUrl(url) {
  return url.split('/').pop() || 'sample.jpg'
}

export default function LiveMonitor({ apiUrl, onLiveResult }) {
  const [mode, setMode] = useState('samples') // 'samples' | 'webcam'
  const [index, setIndex] = useState(0)
  const [result, setResult] = useState(null)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState('')
  const [imageError, setImageError] = useState(false)
  const processingRef = useRef(false)
  const onLiveResultRef = useRef(onLiveResult)
  const nextIndexRef = useRef(1)

  // ---- Laptop webcam state ----
  const [webcamActive, setWebcamActive] = useState(false)
  const [webcamError, setWebcamError] = useState('')
  const [webcamScanning, setWebcamScanning] = useState(false)
  const [webcamResult, setWebcamResult] = useState(null)
  const [webcamCaptureUrl, setWebcamCaptureUrl] = useState(null)
  const [autoScan, setAutoScan] = useState(true)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const webcamBusyRef = useRef(false)

  // Keep latest callback without restarting the interval
  useEffect(() => {
    onLiveResultRef.current = onLiveResult
  }, [onLiveResult])

  // Revoke webcam capture URL on unmount to avoid leaks
  useEffect(() => {
    return () => {
      if (webcamCaptureUrl && webcamCaptureUrl.startsWith('blob:')) {
        try {
          URL.revokeObjectURL(webcamCaptureUrl)
        } catch {
          // ignore
        }
      }
    }
  }, [webcamCaptureUrl])

  useEffect(() => {
    if (mode !== 'samples') return
    let cancelled = false

    const analyzeSample = async (i) => {
      if (cancelled || processingRef.current) return
      processingRef.current = true
      const normalized = ((i % SAMPLE_IMAGES.length) + SAMPLE_IMAGES.length) % SAMPLE_IMAGES.length
      if (!cancelled) {
        setIndex(normalized)
        setScanning(true)
        setError('')
        setImageError(false)
      }
      try {
        const sample = SAMPLE_IMAGES[normalized]
        const imgRes = await fetch(sample.url)
        if (!imgRes.ok) {
          throw new Error(`Sample image not found (${fileNameFromUrl(sample.url)})`)
        }
        const blob = await imgRes.blob()
        const file = new File([blob], fileNameFromUrl(sample.url), {
          type: blob.type || 'image/jpeg',
        })
        const formData = new FormData()
        formData.append('file', file)
        const cam = getCameraById(sample.id)
        formData.append('location', cam ? `${cam.name} (${cam.zone})` : sample.label)
        const res = await axios.post(`${apiUrl}/predict?heatmap=1`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 180000,
        })
        if (cancelled) return
        setResult(res.data)
        onLiveResultRef.current?.({
          cameraId: sample.id,
          imageUrl: sample.url,
          fileName: `${sample.label} — ${fileNameFromUrl(sample.url)}`,
          fire_detected: res.data.fire_detected,
          confidence: res.data.confidence,
          timestamp: res.data.timestamp,
        })
      } catch (err) {
        if (cancelled) return
        setError(err.response?.data?.error || err.message || 'Live prediction failed')
      } finally {
        processingRef.current = false
        if (!cancelled) setScanning(false)
      }
    }

    // Wait for a cold backend to warm up first (bounded, silent — a
    // sleeping free host would otherwise fail every early sample), then
    // analyze the first sample immediately and cycle every CYCLE_MS.
    // nextIndexRef tracks the upcoming camera so a slow /predict never
    // skips a camera and StrictMode remounts can't lose the first sample.
    nextIndexRef.current = 1
    waitForBackend(apiUrl, { timeoutMs: 240000 }).then((ok) => {
      if (cancelled || !ok) return
      analyzeSample(0)
    })
    const id = setInterval(() => {
      if (processingRef.current) return // wait for in-flight request, retry next tick
      const next = nextIndexRef.current % SAMPLE_IMAGES.length
      nextIndexRef.current += 1
      analyzeSample(next)
    }, CYCLE_MS)

    return () => {
      cancelled = true
      processingRef.current = false
      clearInterval(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl, mode])

  const stopWebcam = () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => {
          try {
            t.stop()
          } catch {
            // ignore
          }
        })
        streamRef.current = null
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    } catch {
      // ignore cleanup errors
    }
    setWebcamActive(false)
    setWebcamScanning(false)
    webcamBusyRef.current = false
  }

  // Stop webcam when leaving webcam mode or unmounting
  useEffect(() => {
    if (mode !== 'webcam' && streamRef.current) {
      stopWebcam()
    }
    return () => {
      if (streamRef.current) {
        try {
          streamRef.current.getTracks().forEach((t) => {
            try {
              t.stop()
            } catch {
              // ignore
            }
          })
        } catch {
          // ignore
        }
        streamRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])

  const startWebcam = async () => {
    setWebcamError('')
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setWebcamError('Laptop camera not supported here. Use Chrome/Edge on localhost or HTTPS.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        try {
          await videoRef.current.play()
        } catch {
          // autoplay may need user gesture — video element has controls/muted
        }
      }
      setWebcamActive(true)
    } catch (err) {
      const name = err?.name || ''
      if (name === 'NotAllowedError') {
        setWebcamError('Camera permission denied. Allow camera access in the browser address bar, then try again.')
      } else if (name === 'NotFoundError') {
        setWebcamError('No laptop camera found. Connect a webcam and try again.')
      } else {
        setWebcamError(err?.message || 'Could not start laptop camera.')
      }
    }
  }

  const analyzeWebcamFrame = async () => {
    if (!webcamActive || webcamBusyRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    if (video.readyState < 2 || video.videoWidth === 0) {
      setWebcamError('Camera warming up… press Analyze now again in a second.')
      return
    }
    webcamBusyRef.current = true
    setWebcamScanning(true)
    setWebcamError('')
    try {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92))
      if (!blob) throw new Error('Frame capture failed — try again.')
      const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' })
      const formData = new FormData()
      formData.append('file', file)
      formData.append('location', 'Laptop Webcam')
      const res = await axios.post(`${apiUrl}/predict?heatmap=1`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 180000,
      })
      const previewUrl = URL.createObjectURL(blob)
      setWebcamCaptureUrl((prev) => {
        if (prev && prev.startsWith('blob:')) {
          try {
            URL.revokeObjectURL(prev)
          } catch {
            // ignore
          }
        }
        return previewUrl
      })
      setWebcamResult(res.data)
      onLiveResultRef.current?.({
        imageUrl: previewUrl,
        fileName: 'Laptop Webcam — live capture',
        fire_detected: res.data.fire_detected,
        confidence: res.data.confidence,
        timestamp: res.data.timestamp,
      })
    } catch (err) {
      setWebcamError(err.response?.data?.error || err.message || 'Webcam prediction failed')
    } finally {
      webcamBusyRef.current = false
      setWebcamScanning(false)
    }
  }

  // Auto-scan webcam every WEBCAM_MS while active + enabled
  useEffect(() => {
    if (mode !== 'webcam' || !webcamActive || !autoScan) return undefined
    const id = setInterval(() => {
      if (!webcamBusyRef.current) analyzeWebcamFrame()
    }, WEBCAM_MS)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, webcamActive, autoScan, apiUrl])

  const sample = SAMPLE_IMAGES[index % SAMPLE_IMAGES.length]
  const webcamRisk = webcamResult
    ? getRiskLevel(webcamResult.fire_detected, webcamResult.confidence)
    : null

  return (
    <div className="space-y-5">
      {/* Mode switch */}
      <div className="flex justify-center gap-2 animate-fade-in">
        <button
          type="button"
          onClick={() => setMode('samples')}
          className={`btn-lift px-4 py-2 rounded-xl font-semibold text-xs sm:text-[13px] transition-all active:scale-[0.98] ${
            mode === 'samples' ? 'btn-primary' : 'btn-ghost'
          }`}
        >
          🖼️ Sample feeds
        </button>
        <button
          type="button"
          onClick={() => setMode('webcam')}
          className={`btn-lift px-4 py-2 rounded-xl font-semibold text-xs sm:text-[13px] transition-all active:scale-[0.98] ${
            mode === 'webcam' ? 'btn-primary' : 'btn-ghost'
          }`}
        >
          📷 Laptop camera
        </button>
      </div>

      {mode === 'samples' ? (
        <>
          <div className="glass-panel rounded-2xl shadow-xl shadow-black/30 p-4 sm:p-6 space-y-4 animate-fade-up">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-red-950/70 border border-red-500/50 text-red-200 text-xs sm:text-sm font-bold">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                🔴 LIVE
              </span>
              <span className="text-slate-400 text-xs sm:text-sm">
                {sample.label} • {index + 1} / {SAMPLE_IMAGES.length}
              </span>
            </div>

            <div className="video-frame"><div className="video-frame-inner relative bg-slate-950">
              {!imageError ? (
                <img
                  key={sample.url}
                  src={sample.url}
                  alt={`${sample.label} live feed`}
                  onError={() => setImageError(true)}
                  className="w-full h-64 sm:h-80 object-cover"
                />
              ) : (
                <div className="w-full h-64 sm:h-80 flex flex-col items-center justify-center gap-2 p-6 text-center">
                  <p className="text-4xl">📷</p>
                  <p className="text-slate-300 text-sm font-semibold">
                    Sample image missing: {fileNameFromUrl(sample.url)}
                  </p>
                  <p className="text-slate-500 text-xs">
                    Place 5 images in frontend/public/samples/ (see README there), then refresh.
                  </p>
                </div>
              )}
              {scanning && !imageError && (
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute left-0 right-0 h-12 bg-gradient-to-b from-transparent via-orange-500/30 to-transparent border-y border-orange-400/60 animate-scan" />
                  <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/60 text-orange-300 text-[11px] sm:text-xs font-semibold">
                    Scanning…
                  </span>
                </div>
              )}
            </div></div>

            <div className="flex justify-center gap-1.5">
              {SAMPLE_IMAGES.map((s, i) => (
                <span
                  key={s.url}
                  className={`h-1.5 rounded-full transition-all duration-300 ${
                    i === index % SAMPLE_IMAGES.length
                      ? 'w-6 bg-orange-500'
                      : 'w-1.5 bg-slate-600'
                  }`}
                />
              ))}
            </div>

            <p className="text-center text-slate-500 text-xs">
              Auto-analyzing every 6s • each result is saved to Detection History
            </p>
          </div>

          {error && (
            <div className="bg-red-950/60 border border-red-500/40 text-red-200 px-4 py-3 rounded-xl text-sm sm:text-base transition-all duration-300">
              {error}
            </div>
          )}

          {result && !error && <ResultCard result={result} />}
        </>
      ) : (
        <>
          <div className="glass-panel rounded-2xl shadow-xl shadow-black/30 p-4 sm:p-6 space-y-4 animate-fade-up">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-red-950/70 border border-red-500/50 text-red-200 text-xs sm:text-sm font-bold">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                📷 {webcamActive ? 'WEBCAM LIVE' : 'WEBCAM OFF'}
              </span>
              {webcamRisk && (
                <span
                  className={`px-3 py-1 rounded-full text-xs sm:text-sm font-extrabold ${
                    webcamRisk.key === 'critical' || webcamRisk.key === 'high'
                      ? 'bg-red-600 text-white animate-pulse'
                      : webcamRisk.key === 'safe'
                        ? 'bg-green-600 text-white'
                        : 'bg-yellow-500 text-slate-900'
                  }`}
                >
                  {webcamRisk.icon} {webcamRisk.label}
                </span>
              )}
            </div>

            <div className="video-frame"><div className="video-frame-inner relative bg-slate-950">
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video
                ref={videoRef}
                muted
                playsInline
                className="w-full h-64 sm:h-80 object-cover bg-black"
              />
              <canvas ref={canvasRef} className="hidden" />
              {!webcamActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center bg-slate-950/80">
                  <p className="text-4xl">📷</p>
                  <p className="text-slate-200 text-sm font-semibold">
                    Laptop camera is off
                  </p>
                  <p className="text-slate-400 text-xs max-w-xs">
                    Burn fire in front of your camera (lighter / match / candle at safe distance) and it will alert. Show clear room for SAFE.
                  </p>
                </div>
              )}
              {webcamActive && webcamScanning && (
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute left-0 right-0 h-12 bg-gradient-to-b from-transparent via-orange-500/30 to-transparent border-y border-orange-400/60 animate-scan" />
                  <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/60 text-orange-300 text-[11px] sm:text-xs font-semibold">
                    Analyzing frame…
                  </span>
                </div>
              )}
              {webcamActive && webcamRisk && (webcamRisk.key === 'critical' || webcamRisk.key === 'high') && (
                <div className="absolute bottom-0 left-0 right-0 px-3 py-2 bg-red-600/90 text-white text-center font-extrabold text-sm sm:text-base animate-pulse">
                  🔥 FIRE DETECTED — {((webcamResult.confidence ?? 0) * 100).toFixed(1)}% — alert logged
                </div>
              )}
              {webcamActive && webcamRisk && webcamRisk.key === 'safe' && (
                <div className="absolute bottom-0 left-0 right-0 px-3 py-2 bg-green-600/90 text-white text-center font-bold text-sm">
                  ✅ SAFE — no fire detected
                </div>
              )}
            </div></div>

            <div className="flex flex-wrap justify-center gap-2">
              {!webcamActive ? (
                <button
                  type="button"
                  onClick={startWebcam}
                  className="btn-lift btn-primary px-5 py-2.5 rounded-xl text-[13.5px]"
                >
                  📷 Enable laptop camera
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={analyzeWebcamFrame}
                    disabled={webcamScanning}
                    className="btn-lift btn-primary px-5 py-2.5 rounded-xl text-[13.5px] disabled:opacity-50"
                  >
                    {webcamScanning ? '🔍 Analyzing…' : '🔍 Analyze now'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setAutoScan((v) => !v)}
                    className="btn-lift btn-ghost px-4 py-2.5 rounded-xl text-[13.5px] font-medium"
                  >
                    {autoScan ? '⏸ Auto: ON' : '▶ Auto: OFF'}
                  </button>
                  <button
                    type="button"
                    onClick={stopWebcam}
                    className="btn-lift btn-ghost px-4 py-2.5 rounded-xl text-[13.5px] font-medium"
                  >
                    ⏹ Stop camera
                  </button>
                </>
              )}
            </div>

            <p className="text-center text-slate-500 text-xs">
              {autoScan ? 'Auto-analyzing every 6s' : 'Manual mode — press Analyze now'} • every webcam result is saved to Detection History + triggers critical alert
            </p>
            <p className="text-center text-slate-500 text-[11px]">
              Safety: keep flame small, arm&apos;s length from laptop, ventilated room, adult present.
            </p>
          </div>

          {webcamError && (
            <div className="bg-red-950/60 border border-red-500/40 text-red-200 px-4 py-3 rounded-xl text-sm transition-all duration-300">
              {webcamError}
            </div>
          )}

          {webcamCaptureUrl && (
            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-4 sm:p-5 space-y-2">
              <p className="text-slate-300 text-xs sm:text-sm font-semibold text-center">
                Last analyzed frame
              </p>
              <img
                src={webcamCaptureUrl}
                alt="Last analyzed webcam frame"
                className="w-full max-h-64 object-contain rounded-xl border border-slate-700 bg-slate-950"
              />
            </div>
          )}

          {webcamResult && !webcamError && <ResultCard result={webcamResult} />}
        </>
      )}
    </div>
  )
}
