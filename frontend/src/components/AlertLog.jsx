import { useState } from 'react'

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
 * Permanent session record of all critical alerts (most recent first,
 * capped at 15 in App). Survives toast auto-dismiss; session-only.
 */
export default function AlertLog({ alerts }) {
  const [open, setOpen] = useState(true)

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 transition-all duration-300">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 sm:px-5 py-3 sm:py-4 text-left"
      >
        <span className="font-bold text-sm sm:text-base text-slate-100">
          🚨 Alert Log{' '}
          <span className="text-slate-400 font-semibold">({alerts.length})</span>
        </span>
        <span className="text-slate-400 text-sm">{open ? '▲ Collapse' : '▼ Expand'}</span>
      </button>

      {open && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5">
          {alerts.length === 0 ? (
            <div className="border border-dashed border-slate-600 rounded-xl p-4 text-center">
              <p className="text-slate-400 text-xs sm:text-sm">
                No critical alerts this session.
              </p>
            </div>
          ) : (
            <ul className="space-y-2 max-h-80 overflow-y-auto pr-1">
              {alerts.map((alert) => (
                <li
                  key={alert.id}
                  className="flex items-center gap-3 bg-red-950/40 border border-red-500/30 rounded-xl p-2"
                >
                  <span className="text-xl shrink-0">🚨</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-red-100 text-xs sm:text-sm font-semibold truncate">
                      {alert.location}
                    </p>
                    <p className="text-slate-400 text-[11px] sm:text-xs truncate mt-0.5">
                      {formatTime(alert.timestamp)} •{' '}
                      {((alert.confidence ?? 0) * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                  <span className="px-2 py-0.5 rounded-full bg-red-800 text-white text-[10px] sm:text-xs font-bold shrink-0">
                    CRITICAL
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
