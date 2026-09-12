import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import ResultCard from './ResultCard.jsx'
import { getCameraById } from '../cameraLocations.js'

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

function fileNameFromUrl(url) {
  return url.split('/').pop() || 'sample.jpg'
}

export default function LiveMonitor({ apiUrl, onLiveResult }) {
  const [index, setIndex] = useState(0)
  const [result, setResult] = useState(null)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState('')
  const [imageError, setImageError] = useState(false)
  const processingRef = useRef(false)
  const onLiveResultRef = useRef(onLiveResult)
  const nextIndexRef = useRef(1)

  // Keep latest callback without restarting the interval
  useEffect(() => {
    onLiveResultRef.current = onLiveResult
  }, [onLiveResult])

  useEffect(() => {
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
        const res = await axios.post(`${apiUrl}/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
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

    // Analyze first sample immediately, then cycle every CYCLE_MS.
    // nextIndexRef tracks the upcoming camera so a slow /predict never
    // skips a camera and StrictMode remounts can't lose the first sample.
    nextIndexRef.current = 1
    analyzeSample(0)
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
  }, [apiUrl])

  const sample = SAMPLE_IMAGES[index % SAMPLE_IMAGES.length]

  return (
    <div className="space-y-5">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 p-4 sm:p-6 space-y-4">
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

        <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-950">
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
        </div>

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
    </div>
  )
}
