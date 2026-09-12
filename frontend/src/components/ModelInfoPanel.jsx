import { useEffect, useState } from 'react'
import axios from 'axios'

function formatPct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return `${(Number(v) * 100).toFixed(1)}%`
}

function formatTime(iso) {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  const datePart = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
  const timePart = date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  })
  return `${datePart} at ${timePart}`
}

/**
 * Model evaluation dashboard: test-set accuracy/precision/recall/F1
 * plus a 2x2 confusion matrix (correct = green, errors = red).
 * Fetches GET /model-info; shows a friendly fallback when no training
 * metrics exist yet instead of crashing.
 */
export default function ModelInfoPanel({ apiUrl }) {
  const [open, setOpen] = useState(true)
  const [metrics, setMetrics] = useState(null)
  const [unavailable, setUnavailable] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await axios.get(`${apiUrl}/model-info`)
        if (cancelled) return
        const data = res.data
        if (!data || data.available === false) {
          setUnavailable(true)
        } else {
          setMetrics(data)
        }
      } catch {
        if (!cancelled) setUnavailable(true)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [apiUrl])

  const cm = metrics?.confusion_matrix
  const cmValid =
    Array.isArray(cm) &&
    cm.length === 2 &&
    cm.every((row) => Array.isArray(row) && row.length === 2)
  const [[tn, fp], [fn, tp]] = cmValid ? cm : [[0, 0], [0, 0]]

  const cells = [
    { label: 'TN', value: tn, correct: true, hint: 'Actual No Fire → Predicted No Fire' },
    { label: 'FP', value: fp, correct: false, hint: 'Actual No Fire → Predicted Fire' },
    { label: 'FN', value: fn, correct: false, hint: 'Actual Fire → Predicted No Fire' },
    { label: 'TP', value: tp, correct: true, hint: 'Actual Fire → Predicted Fire' },
  ]

  const statCards = [
    { label: '🎯 Accuracy', value: metrics?.accuracy },
    { label: '🔍 Precision', value: metrics?.precision },
    { label: '📡 Recall', value: metrics?.recall },
    { label: '⚖️ F1-Score', value: metrics?.f1 },
  ]

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 transition-all duration-300">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 sm:px-5 py-3 sm:py-4 text-left"
      >
        <span className="font-bold text-sm sm:text-base text-slate-100">
          🧠 Model Performance
        </span>
        <span className="text-slate-400 text-sm">{open ? '▲ Collapse' : '▼ Expand'}</span>
      </button>

      {open && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5">
          {unavailable || !metrics ? (
            <div className="border border-dashed border-slate-600 rounded-xl p-4 text-center">
              <p className="text-slate-400 text-xs sm:text-sm">
                Run training to see model metrics
              </p>
            </div>
          ) : (
            <>
              {metrics.warning && (
                <div className="bg-amber-950/50 border border-amber-500/40 rounded-xl px-3 py-2 mb-3">
                  <p className="text-amber-200 text-xs sm:text-sm text-center">
                    ⚠️ {metrics.warning}
                  </p>
                </div>
              )}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                {statCards.map((s) => (
                  <div
                    key={s.label}
                    className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center"
                  >
                    <p className="text-slate-400 text-[11px] sm:text-xs">{s.label}</p>
                    <p className="text-slate-100 font-bold text-base sm:text-lg">
                      {formatPct(s.value)}
                    </p>
                  </div>
                ))}
              </div>

              <p className="text-slate-300 text-xs sm:text-sm font-semibold mb-2">
                Confusion Matrix{' '}
                <span className="text-slate-500 font-normal">
                  • rows = actual, columns = predicted
                  {metrics.test_size != null ? ` • n = ${metrics.test_size}` : ''}
                </span>
              </p>
              <div className="max-w-xs mx-auto">
                <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 items-stretch text-center">
                  <span />
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pb-1">
                    Pred No Fire
                  </span>
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pb-1">
                    Pred Fire
                  </span>
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pr-1 self-center">
                    Actual No Fire
                  </span>
                  {cells.slice(0, 2).map((c) => (
                    <div
                      key={c.label}
                      title={c.hint}
                      className={`rounded-xl border p-2 sm:p-3 ${
                        c.correct
                          ? 'bg-green-950/50 border-green-500/40'
                          : 'bg-red-950/50 border-red-500/40'
                      }`}
                    >
                      <p
                        className={`text-[10px] sm:text-xs font-bold ${
                          c.correct ? 'text-green-300' : 'text-red-300'
                        }`}
                      >
                        {c.label}
                      </p>
                      <p className="text-slate-100 font-bold text-base sm:text-lg">
                        {c.value}
                      </p>
                    </div>
                  ))}
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pr-1 self-center">
                    Actual Fire
                  </span>
                  {cells.slice(2).map((c) => (
                    <div
                      key={c.label}
                      title={c.hint}
                      className={`rounded-xl border p-2 sm:p-3 ${
                        c.correct
                          ? 'bg-green-950/50 border-green-500/40'
                          : 'bg-red-950/50 border-red-500/40'
                      }`}
                    >
                      <p
                        className={`text-[10px] sm:text-xs font-bold ${
                          c.correct ? 'text-green-300' : 'text-red-300'
                        }`}
                      >
                        {c.label}
                      </p>
                      <p className="text-slate-100 font-bold text-base sm:text-lg">
                        {c.value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {metrics.trained_at && (
                <p className="text-slate-500 text-xs sm:text-sm text-center mt-3">
                  Trained {formatTime(metrics.trained_at)}
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
