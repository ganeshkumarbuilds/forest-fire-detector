import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import L from 'leaflet'
import { CircleMarker, MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet'
import { CAMERA_LOCATIONS } from '../cameraLocations.js'
import { getRiskLevel, getRiskBadgeClass } from './ResultCard.jsx'

const HOTSPOT_REFRESH_MS = 15 * 60 * 1000

// Marker dot colors per risk level (none = no detection yet at that camera)
const RISK_COLORS = {
  critical: '#dc2626',
  high: '#f97316',
  low: '#eab308',
  safe: '#22c55e',
  none: '#64748b',
}

const iconCache = {}
function markerIcon(riskKey) {
  if (!iconCache[riskKey]) {
    const color = RISK_COLORS[riskKey] ?? RISK_COLORS.none
    iconCache[riskKey] = L.divIcon({
      className: 'risk-marker',
      html: `<div class="risk-dot risk-dot-${riskKey}" style="background:${color}"></div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      popupAnchor: [0, -14],
    })
  }
  return iconCache[riskKey]
}

function formatTime(iso) {
  if (!iso) return 'No detections yet'
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
 * Monitoring network map. `cameraStatus` is keyed by camera id:
 * { 'camera-01': { fire_detected, confidence, timestamp }, ... }
 * Fed by the background simulation + manual live-monitoring results —
 * manual uploads carry no camera, so they never touch the map.
 */

/**
 * Live "Last updated" readout. Own 1s timer so only this line
 * re-renders — the Leaflet map itself is untouched by the ticking.
 */
function LastUpdated({ timestamp }) {
  const [, setNow] = useState(0)

  useEffect(() => {
    if (!timestamp) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [timestamp])

  if (!timestamp) {
    return (
      <p className="text-center text-[11px] sm:text-xs text-slate-500">
        Last updated: waiting for first scan…
      </p>
    )
  }
  const diffSec = Math.max(0, Math.round((Date.now() - new Date(timestamp).getTime()) / 1000))
  const label =
    diffSec < 5 ? 'just now' : diffSec < 60 ? `${diffSec}s ago` : `${Math.floor(diffSec / 60)}m ago`
  return (
    <p className="text-center text-[11px] sm:text-xs text-slate-400">
      Last updated: {label}
    </p>
  )
}

/**
 * Flies between the worldwide camera network and a focused world view
 * when the genuine global satellite layer is toggled. Lives inside MapContainer.
 */
function ViewFlyer({ showGlobal, networkBounds }) {
  const map = useMap()
  useEffect(() => {
    if (showGlobal) {
      map.flyTo([22, 10], 2, { duration: 1.2 })
    } else {
      map.fitBounds(networkBounds)
    }
  }, [showGlobal, map, networkBounds])
  return null
}

export default function MapView({ cameraStatus, apiUrl }) {
  // Worldwide camera network bounds — with cameras on every continent
  // this automatically spans the entire world.
  const bounds = useMemo(
    () => L.latLngBounds(CAMERA_LOCATIONS.map((c) => [c.lat, c.lng])),
    []
  )
  const baseUrl = apiUrl || import.meta.env.VITE_API_URL || 'http://localhost:5000'
  // Genuine worldwide satellite layer (NASA FIRMS). ON by default so the
  // entire world population and nature are protected from the start;
  // toggling flies between world-hotspot view and the camera network.
  const [showGlobal, setShowGlobal] = useState(true)
  const [hotspots, setHotspots] = useState([])
  const [hsMeta, setHsMeta] = useState(null)
  const [hsState, setHsState] = useState('idle') // idle|loading|ready|unavailable

  useEffect(() => {
    if (!showGlobal) return
    let cancelled = false
    const load = async () => {
      setHsState('loading')
      try {
        const res = await axios.get(`${baseUrl}/hotspots`, { timeout: 30000 })
        if (cancelled) return
        const d = res.data ?? {}
        if (d.available && Array.isArray(d.hotspots)) {
          setHotspots(d.hotspots)
          setHsMeta(d)
          setHsState('ready')
        } else {
          setHsState('unavailable')
        }
      } catch {
        if (!cancelled) setHsState('unavailable')
      }
    }
    load()
    const id = setInterval(load, HOTSPOT_REFRESH_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [showGlobal, baseUrl])
  const latestUpdate = useMemo(() => {
    let best = null
    for (const s of Object.values(cameraStatus ?? {})) {
      if (s?.timestamp && (!best || s.timestamp > best)) best = s.timestamp
    }
    return best
  }, [cameraStatus])

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 p-4 sm:p-6 space-y-4 transition-all duration-300">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-bold text-sm sm:text-base text-slate-100">
          🗺️ Worldwide Monitoring Network Map
        </h2>
        <button
          type="button"
          onClick={() => setShowGlobal((v) => !v)}
          title="Show genuine worldwide satellite fire detections (NASA FIRMS)"
          className={`px-3 py-1.5 rounded-lg font-bold text-[11px] sm:text-xs transition-all active:scale-[0.98] ${
            showGlobal
              ? 'text-white bg-gradient-to-r from-orange-600 to-red-600 shadow-lg shadow-red-900/40'
              : 'text-slate-200 bg-slate-700 hover:bg-slate-600 border border-slate-600'
          }`}
        >
          🛰️ {showGlobal ? 'Hide global fires' : 'Global fires'}
        </button>
      </div>

      <MapContainer
        bounds={bounds}
        center={[22, 10]}
        zoom={2}
        minZoom={2}
        maxBounds={[
          [-90, -180],
          [90, 180],
        ]}
        maxBoundsViscosity={1.0}
        worldCopyJump={true}
        scrollWheelZoom={false}
        className="rounded-xl border border-slate-700"
        style={{ height: '380px', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> | Hotspots: <a href="https://firms.modaps.eosdis.nasa.gov/">NASA FIRMS</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          noWrap={false}
        />
        <ViewFlyer showGlobal={showGlobal} networkBounds={bounds} />
        {CAMERA_LOCATIONS.map((cam) => {
          const status = cameraStatus?.[cam.id]
          const riskKey = status ? getRiskLevel(status.fire_detected, status.confidence).key : 'none'
          const risk = status ? getRiskLevel(status.fire_detected, status.confidence) : null
          const pct = status ? ((status.confidence ?? 0) * 100).toFixed(1) : null
          return (
            <Marker key={`${cam.id}-${riskKey}`} position={[cam.lat, cam.lng]} icon={markerIcon(riskKey)}>
              <Popup>
                <div className="text-sm">
                  <p className="font-bold">{cam.name}</p>
                  <p className="opacity-70 text-xs">{cam.zone}</p>
                  {status ? (
                    <div className="mt-1 space-y-1">
                      <span
                        className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-bold ${getRiskBadgeClass(
                          risk.key
                        )}`}
                      >
                        {risk.icon} {risk.label}
                      </span>
                      <p className="text-xs">Confidence: {pct}%</p>
                      <p className="text-xs opacity-70">{formatTime(status.timestamp)}</p>
                    </div>
                  ) : (
                    <p className="text-xs mt-1 opacity-70">
                      No detections yet — start Live Monitoring.
                    </p>
                  )}
                </div>
              </Popup>
            </Marker>
          )
        })}
        {showGlobal &&
          hotspots.map((h, i) => (
            <CircleMarker
              key={`${h.lat},${h.lng},${i}`}
              center={[h.lat, h.lng]}
              radius={5}
              pathOptions={{
                color: '#ff4500',
                weight: 1,
                fillColor: '#ff6a00',
                fillOpacity: 0.75,
              }}
            >
              <Popup>
                <div className="text-sm">
                  <p className="font-bold">🔥 Active fire hotspot</p>
                  <p className="opacity-70 text-xs">
                    {h.lat.toFixed(2)}, {h.lng.toFixed(2)}
                  </p>
                  <p className="text-xs mt-1">
                    {h.satellite || 'Satellite'} • {h.acq_date || ''} {h.acq_time || ''}
                  </p>
                  <p className="text-xs opacity-70">
                    Brightness {h.brightness}K
                    {h.frp ? ` • FRP ${h.frp}MW` : ''}
                    {h.confidence ? ` • ${h.confidence}` : ''}
                  </p>
                  <p className="text-[11px] opacity-60 mt-1">Source: NASA FIRMS satellite data</p>
                </div>
              </Popup>
            </CircleMarker>
          ))}
      </MapContainer>

      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-[11px] sm:text-xs text-slate-400">
        <span>🔴 Critical</span>
        <span>🟠 High risk</span>
        <span>🟡 Low risk</span>
        <span>🟢 Safe</span>
        <span>⚪ No data</span>
        {showGlobal && <span>🔥 NASA hotspot</span>}
      </div>

      {showGlobal && hsState === 'loading' && (
        <p className="text-center text-[11px] sm:text-xs text-slate-400">
          🛰️ Loading live satellite data…
        </p>
      )}
      {showGlobal && hsState === 'ready' && (
        <p className="text-center text-[11px] sm:text-xs text-slate-400">
          🛰️ {hsMeta?.count ?? 0} live hotspots worldwide
          {hsMeta?.total > (hsMeta?.count ?? 0) ? ` (top ${(hsMeta?.count ?? 0)} of ${hsMeta.total})` : ''}{' '}
          • {hsMeta?.source ?? 'NASA FIRMS'}
          {hsMeta?.stale ? ' • cached (offline)' : ''}
        </p>
      )}
      {showGlobal && hsState === 'unavailable' && (
        <p className="text-center text-[11px] sm:text-xs text-slate-500">
          🛰️ Live satellite data unavailable — set FIRMS_MAP_KEY on the server. No
          simulated points are shown.
        </p>
      )}

      <LastUpdated timestamp={latestUpdate} />
    </div>
  )
}
