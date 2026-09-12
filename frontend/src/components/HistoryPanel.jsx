import { useState } from 'react'
import { getRiskLevel, getRiskBadgeClass } from './ResultCard.jsx'

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

export default function HistoryPanel({ history, onClear, stats }) {
  const [open, setOpen] = useState(true)

  // Session Summary — prefers all-time server stats (GET /stats) when loaded,
  // so numbers reflect full history, not just this session. Falls back to
  // computing from the local history array if the backend is unreachable.
  // Note: these are model prediction counts, NOT accuracy (no ground-truth labels exist).
  const total = stats?.total_analyzed ?? history.length
  const fires = stats?.fire_count ?? history.filter((e) => e.fire_detected).length
  const safe = stats?.safe_count ?? history.filter((e) => !e.fire_detected).length
  const avgConfidence =
    stats != null
      ? total === 0
        ? null
        : stats.average_confidence ?? 0
      : total === 0
        ? null
        : history.reduce((sum, e) => sum + (e.confidence ?? 0), 0) / history.length

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 transition-all duration-300">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 sm:px-5 py-3 sm:py-4 text-left"
      >
        <span className="font-bold text-sm sm:text-base text-slate-100">
          📋 Detection History{' '}
          <span className="text-slate-400 font-semibold">({history.length})</span>
        </span>
        <span className="text-slate-400 text-sm">{open ? '▲ Collapse' : '▼ Expand'}</span>
      </button>

      {open && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5">
          <p className="text-slate-300 text-xs sm:text-sm font-semibold mb-2">
            Session Summary{' '}
            <span className="text-slate-500 font-normal">• Model Prediction Stats</span>
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
            <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center">
              <p className="text-slate-400 text-[11px] sm:text-xs">📊 Total Analyzed</p>
              <p className="text-slate-100 font-bold text-base sm:text-lg">{total}</p>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center">
              <p className="text-slate-400 text-[11px] sm:text-xs">🔥 Fire Detected</p>
              <p className="text-red-300 font-bold text-base sm:text-lg">{fires}</p>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center">
              <p className="text-slate-400 text-[11px] sm:text-xs">✅ Safe</p>
              <p className="text-green-300 font-bold text-base sm:text-lg">{safe}</p>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center">
              <p className="text-slate-400 text-[11px] sm:text-xs">📈 Average Confidence</p>
              <p className="text-slate-100 font-bold text-base sm:text-lg">
                {avgConfidence === null ? '—' : `${(avgConfidence * 100).toFixed(1)}%`}
              </p>
            </div>
          </div>

          {history.length === 0 ? (
            <div className="border border-dashed border-slate-600 rounded-xl p-4 text-center">
              <p className="text-slate-400 text-xs sm:text-sm">
                No detections yet — analyze an image or start Live Monitoring.
              </p>
            </div>
          ) : (
            <>
              <div className="flex justify-end mb-2">
                <button
                  type="button"
                  onClick={onClear}
                  className="text-xs text-slate-400 hover:text-red-300 transition-colors"
                >
                  Clear history
                </button>
              </div>
              <ul className="space-y-2 max-h-80 overflow-y-auto pr-1">
                {history.map((entry) => {
                  const risk = getRiskLevel(entry.fire_detected, entry.confidence)
                  const pct = ((entry.confidence ?? 0) * 100).toFixed(1)
                  return (
                    <li
                      key={entry.id}
                      className="flex items-center gap-3 bg-slate-900/50 border border-slate-700/60 rounded-xl p-2"
                    >
                      {entry.imageUrl ? (
                        <img
                          src={entry.imageUrl}
                          alt={entry.fileName || 'detection thumbnail'}
                          className="w-14 h-14 object-cover rounded-lg border border-slate-700 bg-slate-950 shrink-0"
                          loading="lazy"
                        />
                      ) : (
                        <div
                          className="w-14 h-14 rounded-lg border border-slate-700 bg-slate-950 shrink-0 flex items-center justify-center text-2xl"
                          title="No thumbnail stored for this past detection"
                        >
                          🌲
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <span
                          className={`inline-block px-2 py-0.5 rounded-full text-[10px] sm:text-xs font-bold tracking-wide ${getRiskBadgeClass(
                            risk.key
                          )}`}
                        >
                          {risk.icon} {risk.label}
                        </span>
                        <p className="text-slate-400 text-[11px] sm:text-xs truncate mt-1">
                          {entry.fileName || 'image'} • {pct}% • {formatTime(entry.timestamp)}
                        </p>
                      </div>
                      <span className="text-slate-100 font-bold text-xs sm:text-sm shrink-0">
                        {pct}%
                      </span>
                    </li>
                  )
                })}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  )
}
