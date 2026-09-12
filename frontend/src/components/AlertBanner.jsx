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
    second: '2-digit',
  })
  return `${datePart} at ${timePart}`
}

/**
 * Stacked critical-alert toasts. `alerts` is most-recent-first:
 * [{ id, location, riskLabel, confidence, timestamp }, ...]
 */
export default function AlertBanner({ alerts, onDismiss }) {
  if (!alerts || alerts.length === 0) return null

  return (
    <div className="fixed top-4 right-4 z-[9999] w-[calc(100%-2rem)] max-w-sm space-y-2">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          role="alert"
          className="animate-alert-slide-in bg-red-950/95 border border-red-500/50 border-l-4 border-l-red-500 rounded-xl shadow-2xl shadow-red-950/60 p-3 sm:p-4 text-white"
        >
          <div className="flex items-start justify-between gap-2">
            <p className="font-bold text-xs sm:text-sm leading-snug">
              🚨 ALERT: Fire detected at {alert.location} — Notifying Forest Ranger
              Station
            </p>
            <button
              type="button"
              onClick={() => onDismiss(alert.id)}
              aria-label="Dismiss alert"
              className="shrink-0 text-red-300 hover:text-white hover:bg-red-800/60 rounded-lg w-6 h-6 flex items-center justify-center transition-colors"
            >
              ✕
            </button>
          </div>
          <p className="mt-1.5 text-[11px] sm:text-xs text-red-200">
            {alert.riskLabel} • {((alert.confidence ?? 0) * 100).toFixed(1)}% •{' '}
            {formatTime(alert.timestamp)}
          </p>
        </div>
      ))}
    </div>
  )
}
