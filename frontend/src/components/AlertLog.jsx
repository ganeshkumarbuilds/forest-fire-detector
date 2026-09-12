import { useEffect, useState } from 'react'
import axios from 'axios'

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
export default function AlertLog({ alerts, apiUrl }) {
  const [open, setOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [email, setEmail] = useState('')
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')

  const baseUrl = apiUrl || import.meta.env.VITE_API_URL || 'http://localhost:5000'

  // Pre-fill the input with the currently stored email, if set.
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await axios.get(`${baseUrl}/configure-alert-email`, { timeout: 10000 })
        if (!cancelled && res.data?.email) setEmail(res.data.email)
      } catch {
        // Backend unreachable or endpoint missing — stay empty.
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [baseUrl])

  const handleSave = async () => {
    const value = email.trim()
    if (!value) return
    setSaving(true)
    setNotice('')
    try {
      await axios.post(`${baseUrl}/configure-alert-email`, { email: value }, { timeout: 10000 })
      setNotice('✅ Alert email saved.')
      setTimeout(() => setNotice(''), 4000)
    } catch (err) {
      setNotice(err.response?.data?.error || 'Failed to save email.')
    } finally {
      setSaving(false)
    }
  }

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
        <span className="flex items-center gap-2">
          <span
            role="button"
            tabIndex={0}
            title="Email notification settings"
            aria-label="Email notification settings"
            onClick={(e) => {
              e.stopPropagation()
              setSettingsOpen((v) => !v)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                e.stopPropagation()
                setSettingsOpen((v) => !v)
              }
            }}
            className="text-slate-400 hover:text-slate-100 text-base px-1 cursor-pointer"
          >
            ⚙️
          </span>
          <span className="text-slate-400 text-sm">{open ? '▲ Collapse' : '▼ Expand'}</span>
        </span>
      </button>

      {settingsOpen && (
        <div className="mx-4 sm:mx-5 mb-3 rounded-xl border border-slate-700 bg-slate-900/60 p-3 sm:p-4 space-y-2">
          <label
            htmlFor="alert-email-input"
            className="block text-slate-300 text-xs sm:text-sm font-semibold"
          >
            Notify this email on critical fire detections:
          </label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              id="alert-email-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="flex-1 rounded-lg bg-slate-800 border border-slate-600 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-orange-500"
            />
            <button
              type="button"
              onClick={handleSave}
              disabled={saving || !email.trim()}
              className="px-4 py-2 rounded-lg font-bold text-sm text-white bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 disabled:opacity-50 transition-all active:scale-[0.99]"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
          {notice && <p className="text-slate-300 text-xs sm:text-sm">{notice}</p>}
        </div>
      )}

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
