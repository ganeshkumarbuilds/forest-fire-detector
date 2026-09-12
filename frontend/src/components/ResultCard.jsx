/**
 * Shared risk-level mapping.
 * - fire_detected=true  & confidence >= 0.85 → CRITICAL RISK (dark red, flashing)
 * - fire_detected=true  & confidence < 0.85  → HIGH RISK (orange-red)
 * - fire_detected=false & confidence >= 0.85 → SAFE (green)
 * - fire_detected=false & confidence < 0.85  → LOW RISK - MONITOR (yellow)
 */
export function getRiskLevel(fire_detected, confidence) {
  const conf = confidence ?? 0
  if (fire_detected) {
    if (conf >= 0.85) return { key: 'critical', label: 'CRITICAL RISK', icon: '🔥' }
    return { key: 'high', label: 'HIGH RISK', icon: '🔥' }
  }
  if (conf >= 0.85) return { key: 'safe', label: 'SAFE', icon: '✅' }
  return { key: 'low', label: 'LOW RISK - MONITOR', icon: '⚠️' }
}

export function getRiskBadgeClass(key) {
  switch (key) {
    case 'critical':
      return 'bg-red-800 text-white border border-red-500/60 shadow-lg shadow-red-900/50 animate-fire-glow'
    case 'high':
      return 'bg-gradient-to-r from-orange-600 to-red-600 text-white shadow-lg shadow-red-900/40'
    case 'safe':
      return 'bg-green-600 text-white shadow-lg shadow-green-600/30'
    case 'low':
      return 'bg-yellow-500 text-slate-900 shadow-lg shadow-yellow-500/30'
    default:
      return 'bg-slate-600 text-white'
  }
}

export function getRiskBarClass(key) {
  switch (key) {
    case 'critical':
      return 'bg-gradient-to-r from-red-800 to-red-500'
    case 'high':
      return 'bg-gradient-to-r from-orange-500 to-red-600'
    case 'safe':
      return 'bg-gradient-to-r from-green-500 to-emerald-500'
    case 'low':
      return 'bg-gradient-to-r from-yellow-400 to-amber-500'
    default:
      return 'bg-slate-500'
  }
}

export default function ResultCard({ result }) {
  const { fire_detected, confidence, timestamp, heatmap_image } = result
  // Always clean to 1 decimal place, e.g. 87.3% (100% shows as 100.0%)
  const pct = ((confidence ?? 0) * 100).toFixed(1)
  const barWidth = `${Math.min(100, Math.max(0, (confidence ?? 0) * 100))}%`
  const risk = getRiskLevel(fire_detected, confidence)

  // Format raw ISO string to a clean local-time string:
  // e.g. "Analyzed on Sep 11, 2026 at 4:30 PM"
  const formatTimestamp = (iso) => {
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
    return `Analyzed on ${datePart} at ${timePart}`
  }

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 p-5 sm:p-6 space-y-4 animate-fade-in transition-all duration-300">
      <div className="flex justify-center">
        <span
          className={`px-6 py-3 rounded-full text-base sm:text-xl font-extrabold tracking-wide transition-all duration-300 flex items-center gap-2 ${getRiskBadgeClass(
            risk.key
          )}`}
        >
          {risk.key === 'critical' && (
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
              <span className="relative inline-flex rounded-full h-3 w-3 bg-white" />
            </span>
          )}
          {risk.icon} {risk.label}
        </span>
      </div>

      <div>
        <div className="flex justify-between items-baseline mb-1">
          <p className="text-slate-300 text-sm sm:text-base">Confidence</p>
          <p className="text-slate-100 font-bold text-sm sm:text-base">{pct}%</p>
        </div>
        <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${getRiskBarClass(
              risk.key
            )}`}
            style={{ width: barWidth }}
          />
        </div>
      </div>

      {timestamp && (
        <p className="text-slate-500 text-xs sm:text-sm text-center">
          {formatTimestamp(timestamp)}
        </p>
      )}

      {heatmap_image && (
        <details className="bg-slate-900/60 border border-slate-700 rounded-xl overflow-hidden">
          <summary className="cursor-pointer px-4 py-3 text-sm sm:text-base font-semibold text-orange-300 hover:text-orange-200 hover:bg-slate-700/40 transition-colors list-none text-center">
            Show Model Focus Area
          </summary>
          <div className="px-4 pb-4">
            <img
              src={`data:image/png;base64,${heatmap_image}`}
              alt="Grad-CAM heatmap showing the areas the model focused on"
              className="w-full max-h-80 object-contain rounded-lg border border-slate-700 shadow-lg shadow-black/40 bg-slate-950"
            />
            <p className="text-slate-400 text-xs sm:text-sm text-center mt-2">
              Red/yellow areas show what the model focused on to make this prediction.
            </p>
          </div>
        </details>
      )}
    </div>
  )
}
