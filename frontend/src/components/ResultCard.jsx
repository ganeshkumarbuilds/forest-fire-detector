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
  const { fire_detected, confidence, prob_fire, prob_no_fire, predicted_class, timestamp, heatmap_image, original_image, quality } = result
  // Always clean to 1 decimal place, e.g. 87.3% (100% shows as 100.0%)
  const pct = ((confidence ?? 0) * 100).toFixed(1)
  const barWidth = `${Math.min(100, Math.max(0, (confidence ?? 0) * 100))}%`
  const risk = getRiskLevel(fire_detected, confidence)
  
  // Class probabilities (with safe fallbacks)
  const firePct = ((prob_fire ?? (fire_detected ? confidence : 1 - confidence)) * 100).toFixed(1)
  const noFirePct = ((prob_no_fire ?? (fire_detected ? 1 - confidence : confidence)) * 100).toFixed(1)
  
  // Predicted class label (with fallback)
  const classLabel = predicted_class ?? (fire_detected ? 'FIRE' : 'NO FIRE')

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

  // Handle missing/failed heatmap gracefully
  const hasHeatmap = Boolean(heatmap_image)

  return (
    <div className="glass-panel rounded-2xl shadow-xl shadow-black/30 p-5 sm:p-6 space-y-4 animate-fade-up transition-all duration-300">
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

      {/* Predicted Class */}
      <div className="text-center">
        <p className="text-slate-300 text-sm sm:text-base mb-1">Predicted Class</p>
        <p className={`text-2xl sm:text-3xl font-extrabold tracking-wide ${fire_detected ? 'text-red-400' : 'text-green-400'}`}>
          {classLabel}
        </p>
      </div>

      {/* Class Probabilities */}
      <div className="space-y-3">
        <p className="text-slate-300 text-sm sm:text-base text-center">Class Probabilities</p>
        
        {/* Visual comparison bar */}
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
          <div className="space-y-3">
            {/* FIRE probability bar */}
            <div>
              <div className="flex justify-between items-baseline mb-1">
                <span className="text-red-400 text-sm font-medium flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-red-400"></span>
                  FIRE
                </span>
                <span className="text-slate-100 font-bold text-sm">{firePct}%</span>
              </div>
              <div className="w-full h-4 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-700 ease-out"
                  style={{ width: `${Math.min(100, Math.max(0, parseFloat(firePct)))}%` }}
                />
              </div>
            </div>
            
            {/* NO FIRE probability bar */}
            <div>
              <div className="flex justify-between items-baseline mb-1">
                <span className="text-green-400 text-sm font-medium flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-green-400"></span>
                  NO FIRE
                </span>
                <span className="text-slate-100 font-bold text-sm">{noFirePct}%</span>
              </div>
              <div className="w-full h-4 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-green-600 to-green-400 transition-all duration-700 ease-out"
                  style={{ width: `${Math.min(100, Math.max(0, parseFloat(noFirePct)))}%` }}
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Compact probability cards */}
        <div className="grid grid-cols-2 gap-3 sm:gap-4">
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3 sm:p-4 text-center">
            <p className="text-slate-400 text-xs sm:text-sm font-medium mb-1">FIRE</p>
            <p className="text-red-400 text-xl sm:text-2xl font-bold">{firePct}%</p>
          </div>
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3 sm:p-4 text-center">
            <p className="text-slate-400 text-xs sm:text-sm font-medium mb-1">NO FIRE</p>
            <p className="text-green-400 text-xl sm:text-2xl font-bold">{noFirePct}%</p>
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div>
        <div className="flex justify-between items-baseline mb-1">
          <p className="text-slate-300 text-sm sm:text-base">Confidence</p>
          <p className="text-slate-100 font-bold text-sm sm:text-base">{pct}%</p>
        </div>
        <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden bar-shine">
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

      {/* Image Quality Section */}
      {quality && (
        <div className="space-y-3 pt-2 border-t border-slate-700/50">
          <div className="flex items-center justify-between">
            <h3 className="text-slate-300 text-sm sm:text-base font-semibold flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${quality.valid ? 'bg-green-400' : 'bg-red-400'}`}></span>
              Image Quality
            </h3>
            <span className={`text-xs px-2 py-1 rounded ${quality.valid ? 'bg-green-950/50 text-green-300' : 'bg-red-950/50 text-red-300'}`}>
              {quality.valid ? 'Passed' : 'Failed'}
            </span>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center">
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
              <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Dimensions</p>
              <p className="text-slate-100 font-mono text-sm sm:text-base">
                {quality.width} × {quality.height}
              </p>
            </div>
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
              <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Blur Score</p>
              <p className="text-slate-100 font-mono text-sm sm:text-base">
                {quality.blur_score?.toFixed?.(1) ?? '—'}
              </p>
              <p className="text-[10px] text-slate-500 mt-1">(higher = sharper)</p>
            </div>
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
              <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Brightness</p>
              <p className="text-slate-100 font-mono text-sm sm:text-base">
                {quality.brightness?.toFixed?.(1) ?? '—'} / 255
              </p>
            </div>
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-3">
              <p className="text-slate-400 text-[11px] sm:text-xs font-medium mb-1">Contrast</p>
              <p className="text-slate-100 font-mono text-sm sm:text-base">
                {quality.contrast?.toFixed?.(1) ?? '—'}
              </p>
            </div>
          </div>

          {quality.warnings && quality.warnings.length > 0 && (
            <div className="bg-amber-950/50 border border-amber-500/40 rounded-xl p-3">
              <p className="text-amber-200 text-xs font-medium mb-1 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                Quality Warnings
              </p>
              <ul className="text-amber-100 text-xs space-y-1">
                {quality.warnings.map((w, i) => (
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
        </div>
      )}

      {/* AI Explanation / Model Attention Section */}
      <div className="space-y-3 pt-2 border-t border-slate-700/50">
        <div className="flex items-center justify-between">
          <h3 className="text-slate-300 text-sm sm:text-base font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-orange-400"></span>
            AI Explanation — Model Attention
          </h3>
          {!hasHeatmap && (
            <span className="text-slate-500 text-xs px-2 py-1 rounded bg-slate-800/60">
              Not available
            </span>
          )}
        </div>
        <p className="text-slate-500 text-xs sm:text-sm">
          The heatmap highlights image regions that most influenced the model&apos;s prediction.
          Red/yellow areas indicate higher attention; blue/green areas indicate lower attention.
          This is generated using FD-CAM (Finite-Difference Class Activation Mapping) on the
          model&apos;s final convolutional layer.
        </p>

        {hasHeatmap ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Original Image */}
            {original_image && (
              <div className="bg-slate-900/60 border border-slate-700 rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-slate-800/60 border-b border-slate-700 text-center">
                  <p className="text-slate-400 text-xs font-medium uppercase tracking-wide">Original Image</p>
                </div>
                <div className="aspect-square relative">
                  <img
                    src={original_image}
                    alt="Original uploaded image"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            )}

            {/* FD-CAM Heatmap */}
            <div className="bg-slate-900/60 border border-slate-700 rounded-xl overflow-hidden">
              <div className="px-3 py-2 bg-slate-800/60 border-b border-slate-700 text-center">
                <p className="text-orange-400 text-xs font-medium uppercase tracking-wide flex items-center justify-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-orange-400"></span>
                  Model Attention (FD-CAM)
                </p>
              </div>
              <div className="aspect-square relative p-2">
                <img
                  src={`data:image/png;base64,${heatmap_image}`}
                  alt="FD-CAM heatmap showing the areas the model focused on for this prediction"
                  className="w-full h-full object-contain rounded-lg"
                />
              </div>
              <div className="px-3 pb-3">
                <p className="text-slate-500 text-xs text-center">
                  Red/yellow = high attention &nbsp;•&nbsp; Blue/green = low attention
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-6 text-center">
            <div className="text-slate-500 text-sm mb-2">No attention map available</div>
            <p className="text-slate-600 text-xs">
              The model did not return a heatmap for this prediction. This can happen if
              the model is still warming up or if the heatmap generation timed out.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
