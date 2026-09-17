import { useEffect, useState } from 'react'
import axios from 'axios'

function formatPct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return `${(Number(v) * 100).toFixed(1)}%`
}
function formatProb(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(4)
}

export default function ErrorAnalysisPanel({ apiUrl }) {
  const [open, setOpen] = useState(true)
  const [data, setData] = useState(null)
  const [unavailable, setUnavailable] = useState(false)
  const [showExplanations, setShowExplanations] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await axios.get(`${apiUrl}/error-analysis`)
        if (cancelled) return
        const d = res.data
        if (!d || d.available === false) {
          setUnavailable(true)
        } else {
          setData(d)
        }
      } catch {
        if (!cancelled) setUnavailable(true)
      }
    }
    load()
    return () => { cancelled = true }
  }, [apiUrl])

  const agg = data?.aggregate_metrics
  const split = data?.split
  const dataset = data?.dataset
  const meta = data?.evaluation_metadata
  const stats = data?.descriptive_error_statistics
  const fps = data?.false_positives || []
  const fns = data?.false_negatives || []
  const fpExplanations = data?.fp_explanations || []
  const explanationsAvailable = data?.explanations_available === true && fpExplanations.length > 0

  const cm = agg?.confusion_matrix
  const cmValid = Array.isArray(cm) && cm.length === 2 && cm.every(r => Array.isArray(r) && r.length === 2)
  const [[tn, fp], [fn, tp]] = cmValid ? cm : [[0,0],[0,0]]

  const cells = [
    { label: 'TN', value: tn, correct: true, hint: 'Actual No Fire → Predicted No Fire' },
    { label: 'FP', value: fp, correct: false, hint: 'Actual No Fire → Predicted Fire' },
    { label: 'FN', value: fn, correct: false, hint: 'Actual Fire → Predicted No Fire' },
    { label: 'TP', value: tp, correct: true, hint: 'Actual Fire → Predicted Fire' },
  ]

  const statCards = agg ? [
    { label: '🎯 Accuracy', value: agg.accuracy },
    { label: '🔍 Precision', value: agg.precision },
    { label: '📡 Recall', value: agg.recall },
    { label: '⚖️ F1-Score', value: agg.f1 },
  ] : []

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 transition-all duration-300">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 sm:px-5 py-3 sm:py-4 text-left"
      >
        <span className="font-bold text-sm sm:text-base text-slate-100">
          🔎 Error Analysis
        </span>
        <span className="text-slate-400 text-sm">{open ? '▲ Collapse' : '▼ Expand'}</span>
      </button>

      {open && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5">
          {unavailable || !data ? (
            <div className="border border-dashed border-slate-600 rounded-xl p-4 text-center">
              <p className="text-slate-400 text-xs sm:text-sm">No error analysis available. Run <span className="font-mono">ml-model/error_analysis.py</span> to generate <span className="font-mono">error_analysis.json</span>.</p>
            </div>
          ) : (
            <>
              {/* Measured evaluation results */}
              <p className="text-slate-500 text-[11px] uppercase tracking-wide font-semibold mb-2">Measured evaluation results</p>
              <div className="bg-slate-900/40 border border-slate-700/60 rounded-xl p-3 mb-3">
                <div className="grid grid-cols-3 gap-2 text-center text-xs sm:text-sm">
                  <div><p className="text-slate-500">Test size</p><p className="text-slate-100 font-bold text-base">{agg?.test_size ?? '—'}</p></div>
                  <div><p className="text-slate-500">FP</p><p className="text-red-300 font-bold text-base">{agg?.fp ?? 0}</p></div>
                  <div><p className="text-slate-500">FN</p><p className="text-red-300 font-bold text-base">{agg?.fn ?? 0}</p></div>
                </div>
                <p className="text-slate-500 text-[11px] text-center mt-2">
                  Dataset {dataset?.total_images ?? '—'} images • fire {dataset?.fire_count ?? '—'} / no_fire {dataset?.no_fire_count ?? '—'} • split 70/15/15 stratified random_state {split?.random_state ?? 42} • threshold {meta?.decision_threshold ?? 0.5}
                </p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                {statCards.map(s => (
                  <div key={s.label} className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center">
                    <p className="text-slate-400 text-[11px] sm:text-xs">{s.label}</p>
                    <p className="text-slate-100 font-bold text-base sm:text-lg">{formatPct(s.value)}</p>
                  </div>
                ))}
              </div>

              <p className="text-slate-300 text-xs sm:text-sm font-semibold mb-2">
                Confusion Matrix <span className="text-slate-500 font-normal">• rows = actual, columns = predicted • n = {agg?.test_size}</span>
              </p>
              <div className="max-w-xs mx-auto mb-4">
                <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 items-stretch text-center">
                  <span />
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pb-1">Pred No Fire</span>
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pb-1">Pred Fire</span>
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pr-1 self-center">Actual No Fire</span>
                  {cells.slice(0,2).map(c => (
                    <div key={c.label} title={c.hint} className={`rounded-xl border p-2 sm:p-3 ${c.correct ? 'bg-green-950/50 border-green-500/40' : 'bg-red-950/50 border-red-500/40'}`}>
                      <p className={`text-[10px] sm:text-xs font-bold ${c.correct ? 'text-green-300' : 'text-red-300'}`}>{c.label}</p>
                      <p className="text-slate-100 font-bold text-base sm:text-lg">{c.value}</p>
                    </div>
                  ))}
                  <span className="text-slate-500 text-[11px] sm:text-xs font-semibold pr-1 self-center">Actual Fire</span>
                  {cells.slice(2).map(c => (
                    <div key={c.label} title={c.hint} className={`rounded-xl border p-2 sm:p-3 ${c.correct ? 'bg-green-950/50 border-green-500/40' : 'bg-red-950/50 border-red-500/40'}`}>
                      <p className={`text-[10px] sm:text-xs font-bold ${c.correct ? 'text-green-300' : 'text-red-300'}`}>{c.label}</p>
                      <p className="text-slate-100 font-bold text-base sm:text-lg">{c.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Descriptive observed patterns */}
              {stats && (
                <div className="bg-slate-900/40 border border-slate-700/60 rounded-xl p-3 mb-4">
                  <p className="text-slate-500 text-[11px] uppercase tracking-wide font-semibold mb-1">Descriptive observed patterns</p>
                  <p className="text-slate-400 text-xs leading-relaxed">
                    {stats.fp_count} of {agg?.fp ?? 0} false positives had a quality warning: {stats.fp_with_quality_warnings} with warnings {stats.fp_count > 0 ? `(${(stats.fp_with_quality_warnings/stats.fp_count*100).toFixed(0)}%)` : ''}.
                    {' '}{stats.fn_count} of {agg?.fn ?? 0} false negatives had a quality warning: {stats.fn_with_quality_warnings} with warnings.
                  </p>
                  {stats.fp_prob_fire_stats?.min != null && (
                    <p className="text-slate-500 text-xs mt-1">FP prob_fire: min {formatProb(stats.fp_prob_fire_stats.min)} • max {formatProb(stats.fp_prob_fire_stats.max)} • mean {formatProb(stats.fp_prob_fire_stats.mean)} • median {formatProb(stats.fp_prob_fire_stats.median)}</p>
                  )}
                  {Object.keys(stats.fp_warning_types || {}).length > 0 && (
                    <p className="text-slate-500 text-xs mt-1">FP warning breakdown: {Object.entries(stats.fp_warning_types).map(([k,v]) => `${v}× ${k}`).join(' • ')}</p>
                  )}
                  {Object.keys(stats.fn_warning_types || {}).length > 0 && (
                    <p className="text-slate-500 text-xs mt-1">FN warning breakdown: {Object.entries(stats.fn_warning_types).map(([k,v]) => `${v}× ${k}`).join(' • ')}</p>
                  )}
                  <p className="text-amber-400/70 text-[11px] mt-2 italic">Observed associations only — do not imply causation. E.g., low contrast was observed in some errors but not proven to cause them.</p>
                </div>
              )}

              {/* False Positives */}
              <p className="text-slate-300 text-xs sm:text-sm font-semibold mb-2">False Positives — No Fire predicted as Fire <span className="text-slate-500 font-normal">• n={fps.length}</span></p>
              {fps.length === 0 ? (
                <p className="text-slate-400 text-sm bg-slate-900/50 border border-slate-700/60 rounded-xl p-3 text-center mb-4">No false positives in this test set</p>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-slate-700/60 mb-4">
                  <table className="w-full text-xs sm:text-sm">
                    <thead className="bg-slate-900/70 text-slate-400">
                      <tr>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">Filename</th>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">True</th>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">Pred</th>
                        <th className="px-2 sm:px-3 py-2 text-right font-semibold">P(fire)</th>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">Quality warnings</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/60">
                      {fps.map(r => (
                        <tr key={r.filename} className="bg-slate-800/40">
                          <td className="px-2 sm:px-3 py-2 font-mono text-[11px] sm:text-xs text-slate-200 break-all">{r.filename}</td>
                          <td className="px-2 sm:px-3 py-2 text-slate-400">{r.true_class}</td>
                          <td className="px-2 sm:px-3 py-2"><span className="px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 text-[11px] font-bold">{r.predicted_class}</span></td>
                          <td className="px-2 sm:px-3 py-2 text-right font-mono text-slate-100">{formatProb(r.prob_fire)}</td>
                          <td className="px-2 sm:px-3 py-2 text-slate-400 text-[11px]">{r.quality?.warnings?.length ? r.quality.warnings.join(' • ') : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Explainable Error Investigation (offline FD-CAM) */}
              <div className="bg-slate-900/40 border border-slate-700/60 rounded-xl mb-4">
                <button
                  type="button"
                  onClick={() => setShowExplanations(v => !v)}
                  className="w-full flex items-center justify-between px-3 py-2.5 text-left"
                >
                  <span className="text-slate-200 text-xs sm:text-sm font-semibold">
                    Explainable Error Investigation — FD-CAM <span className="text-slate-500 font-normal">• n={fpExplanations.length}</span>
                  </span>
                  <span className="text-slate-400 text-xs">{showExplanations ? '▲ Collapse' : '▼ Expand'}</span>
                </button>
                {showExplanations && (
                  <div className="px-3 pb-3">
                    <p className="text-slate-400 text-[11px] sm:text-xs leading-relaxed mb-3">
                      FD-CAM highlights image regions associated with the model&apos;s predicted class. It does not prove why the prediction was incorrect or establish causation.
                    </p>
                    {!explanationsAvailable ? (
                      <p className="text-slate-500 text-xs text-center border border-dashed border-slate-700 rounded-xl p-3">Explanations unavailable.</p>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {fpExplanations.map(e => (
                          <details key={e.filename} className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-3">
                            <summary className="cursor-pointer text-slate-200 font-mono text-[11px] sm:text-xs break-all">{e.filename}</summary>
                            <div className="flex flex-wrap items-center gap-1.5 mt-2 text-[11px]">
                              <span className="text-slate-400">True: No Fire</span>
                              <span className="px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 font-bold">Predicted: Fire</span>
                              <span className="text-slate-300 font-mono ml-auto">P(fire) {formatProb(e.prob_fire)}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 mt-2">
                              <div>
                                <p className="text-slate-500 text-[11px] font-semibold mb-1">Original</p>
                                {e.original_image ? (
                                  <img src={`data:image/png;base64,${e.original_image}`} alt={`Original ${e.filename}`} className="w-full rounded-lg border border-slate-700/60" loading="lazy" />
                                ) : (
                                  <p className="text-slate-500 text-[11px]">Original unavailable</p>
                                )}
                              </div>
                              <div>
                                <p className="text-slate-500 text-[11px] font-semibold mb-1">FD-CAM overlay</p>
                                {e.heatmap_image ? (
                                  <img src={`data:image/png;base64,${e.heatmap_image}`} alt={`FD-CAM ${e.filename}`} className="w-full rounded-lg border border-slate-700/60" loading="lazy" />
                                ) : (
                                  <p className="text-slate-500 text-[11px]">Heatmap unavailable</p>
                                )}
                              </div>
                            </div>
                          </details>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* False Negatives */}
              <p className="text-slate-300 text-xs sm:text-sm font-semibold mb-2">False Negatives — Fire predicted as No Fire <span className="text-slate-500 font-normal">• n={fns.length}</span></p>
              {fns.length === 0 ? (
                <p className="text-slate-400 text-sm bg-slate-900/50 border border-slate-700/60 rounded-xl p-3 text-center">No false negatives in this test set</p>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-slate-700/60">
                  <table className="w-full text-xs sm:text-sm">
                    <thead className="bg-slate-900/70 text-slate-400">
                      <tr>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">Filename</th>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">True</th>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">Pred</th>
                        <th className="px-2 sm:px-3 py-2 text-right font-semibold">P(fire)</th>
                        <th className="px-2 sm:px-3 py-2 text-left font-semibold">Quality warnings</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/60">
                      {fns.map(r => (
                        <tr key={r.filename} className="bg-slate-800/40">
                          <td className="px-2 sm:px-3 py-2 font-mono text-[11px] sm:text-xs text-slate-200 break-all">{r.filename}</td>
                          <td className="px-2 sm:px-3 py-2 text-slate-400">{r.true_class}</td>
                          <td className="px-2 sm:px-3 py-2"><span className="px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-300 text-[11px] font-bold">{r.predicted_class}</span></td>
                          <td className="px-2 sm:px-3 py-2 text-right font-mono text-slate-100">{formatProb(r.prob_fire)}</td>
                          <td className="px-2 sm:px-3 py-2 text-slate-400 text-[11px]">{r.quality?.warnings?.length ? r.quality.warnings.join(' • ') : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Hypotheses disclaimer */}
              <div className="mt-4 bg-amber-950/20 border border-amber-500/20 rounded-xl p-3">
                <p className="text-amber-300 text-xs font-semibold">Interpretation guidance</p>
                <p className="text-slate-400 text-xs mt-1 leading-relaxed">
                  <span className="text-slate-300 font-medium">Measured results:</span> metrics and confusion matrix above.
                  {' '}<span className="text-slate-300 font-medium">Observed patterns:</span> descriptive statistics (e.g., warning frequencies) — correlation only.
                  {' '}<span className="text-slate-300 font-medium">Hypotheses:</span> possible explanations (e.g., reddish smoke misleads model) are not proven and require further controlled experiments.
                </p>
                {meta?.timestamp && <p className="text-slate-500 text-[11px] mt-2">Evaluated {new Date(meta.timestamp).toLocaleString()} • threshold {meta.decision_threshold}</p>}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
