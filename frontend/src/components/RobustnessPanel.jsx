import { useEffect, useState } from 'react'
import axios from 'axios'

function formatPct(v) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return `${(Number(v)*100).toFixed(1)}%`
}
function formatDelta(v) {
  if (v == null) return '—'
  const s = v>0 ? '+' : ''
  return `${s}${(Number(v)*100).toFixed(1)}pp`
}

export default function RobustnessPanel({ apiUrl }) {
  const [open, setOpen] = useState(true)
  const [data, setData] = useState(null)
  const [unavailable, setUnavailable] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await axios.get(`${apiUrl}/robustness`)
        if (cancelled) return
        if (!res.data || res.data.available === false) setUnavailable(true)
        else setData(res.data)
      } catch { if (!cancelled) setUnavailable(true) }
    }
    load()
    return () => { cancelled = true }
  }, [apiUrl])

  const clean = data?.clean_metrics
  const perts = data?.perturbations || []
  const meta = data?.metadata
  const split = data?.split

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 transition-all duration-300">
      <button type="button" onClick={()=>setOpen(v=>!v)} className="w-full flex items-center justify-between px-4 sm:px-5 py-3 sm:py-4 text-left">
        <span className="font-bold text-sm sm:text-base text-slate-100">🧪 Robustness Testing</span>
        <span className="text-slate-400 text-sm">{open ? '▲ Collapse' : '▼ Expand'}</span>
      </button>
      {open && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5">
          {unavailable || !data ? (
            <div className="border border-dashed border-slate-600 rounded-xl p-4 text-center">
              <p className="text-slate-400 text-xs sm:text-sm">No robustness results. Run <span className="font-mono">ml-model/robustness_test.py</span> to generate <span className="font-mono">robustness.json</span>.</p>
            </div>
          ) : (
            <>
              <p className="text-slate-500 text-[11px] uppercase tracking-wide font-semibold mb-2">Clean baseline (measured)</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                {[
                  {label:'🎯 Accuracy', v:clean?.accuracy},
                  {label:'🔍 Precision', v:clean?.precision},
                  {label:'📡 Recall', v:clean?.recall},
                  {label:'⚖️ F1', v:clean?.f1},
                ].map(c=>(
                  <div key={c.label} className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-2 sm:p-3 text-center">
                    <p className="text-slate-400 text-[11px]">{c.label}</p>
                    <p className="text-slate-100 font-bold text-base">{formatPct(c.v)}</p>
                  </div>
                ))}
              </div>
              <p className="text-slate-500 text-xs text-center mb-3">n={clean?.test_size} • FP {clean?.fp} FN {clean?.fn} • threshold {meta?.threshold} • seed {meta?.random_seed}</p>

              <p className="text-slate-300 text-xs sm:text-sm font-semibold mb-2">Perturbations — controlled experiments</p>
              <div className="overflow-x-auto rounded-xl border border-slate-700/60 mb-3">
                <table className="w-full text-xs">
                  <thead className="bg-slate-900/70 text-slate-400">
                    <tr>
                      <th className="px-2 py-2 text-left">Perturbation</th>
                      <th className="px-2 py-2 text-left">Level</th>
                      <th className="px-2 py-2 text-right">Acc</th>
                      <th className="px-2 py-2 text-right">Δ Acc</th>
                      <th className="px-2 py-2 text-right">Change%</th>
                      <th className="px-2 py-2 text-right">Avg |Δp|</th>
                      <th className="px-2 py-2 text-right">FP→</th>
                      <th className="px-2 py-2 text-right">FN→</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/60">
                    {perts.map(p=>{
                      const deltaColor = p.metric_deltas.accuracy_delta===0 ? 'text-slate-300' : p.metric_deltas.accuracy_delta>0 ? 'text-green-300' : 'text-red-300'
                      return (
                        <tr key={`${p.name}-${p.level}`} className="bg-slate-800/40">
                          <td className="px-2 py-2 font-medium text-slate-200">{p.name}</td>
                          <td className="px-2 py-2 font-mono text-slate-400">{p.level}</td>
                          <td className="px-2 py-2 text-right font-mono text-slate-100">{formatPct(p.metrics.accuracy)}</td>
                          <td className={`px-2 py-2 text-right font-mono ${deltaColor}`}>{formatDelta(p.metric_deltas.accuracy_delta)}</td>
                          <td className="px-2 py-2 text-right font-mono text-amber-200">{(p.prediction_change_rate*100).toFixed(1)}%</td>
                          <td className="px-2 py-2 text-right font-mono text-slate-300">{p.avg_absolute_probability_change.toFixed(4)}</td>
                          <td className="px-2 py-2 text-right font-mono text-slate-200">{clean.fp}→{p.metrics.fp} <span className={p.metric_deltas.fp_delta>0?'text-red-300':'text-green-300'}>({p.metric_deltas.fp_delta>0?'+':''}{p.metric_deltas.fp_delta})</span></td>
                          <td className="px-2 py-2 text-right font-mono text-slate-200">{clean.fn}→{p.metrics.fn} <span className={p.metric_deltas.fn_delta>0?'text-red-300':'text-slate-400'}>({p.metric_deltas.fn_delta>0?'+':''}{p.metric_deltas.fn_delta})</span></td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              <p className="text-slate-300 text-xs font-semibold mb-1">Clean false positives under perturbation</p>
              <p className="text-slate-500 text-[11px] mb-2">“X of 6 clean false positives changed class under …” — observed, not causal.</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
                {perts.map(p=>(
                  <div key={p.name+p.level} className="bg-slate-900/40 border border-slate-700/60 rounded-lg px-3 py-2">
                    <p className="text-slate-400 text-[11px] font-mono">{p.name} {p.level}</p>
                    <p className="text-slate-200 text-xs">{p.clean_fp_stability.changed} of {p.clean_fp_stability.total_clean_fp} clean false positives changed class under {p.name} {p.level}.</p>
                  </div>
                ))}
              </div>

              <div className="bg-amber-950/20 border border-amber-500/20 rounded-xl p-3">
                <p className="text-amber-300 text-xs font-semibold">Methodological note</p>
                <p className="text-slate-400 text-xs mt-1 leading-relaxed">
                  Synthetic perturbations are controlled experiments and do not constitute proof of real-world robustness. Results use the fixed held-out test set n={split?.test_size} (70/15/15 split, random_state {split?.random_state}), threshold 0.5, no retraining. Robustness depends on perturbation severity; this is not an OOD benchmark and does not represent real distribution shifts.
                </p>
                {meta?.evaluation_timestamp && <p className="text-slate-500 text-[11px] mt-2">Evaluated {new Date(meta.evaluation_timestamp).toLocaleString()}</p>}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
