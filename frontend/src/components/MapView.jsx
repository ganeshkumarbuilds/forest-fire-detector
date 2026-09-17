import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import L from 'leaflet'
import { CircleMarker, LayersControl, MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet'
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

// Genuine hotspot color by satellite-measured brightness — hotter = redder
function hotspotStyle(h) {
  const b = Number(h.brightness) || 0
  const c = Number(h.confidence === 'h' || h.confidence === 'high' ? 1 : h.confidence === 'n' || h.confidence === 'nominal' ? 0.5 : 0) // keep brightness primary
  void c
  if (b >= 360) return { color: '#991b1b', fillColor: '#dc2626', fillOpacity: 0.85, weight: 1.2 }
  if (b >= 330) return { color: '#c2410c', fillColor: '#f97316', fillOpacity: 0.78, weight: 1 }
  if (b >= 310) return { color: '#a16207', fillColor: '#f59e0b', fillOpacity: 0.72, weight: 1 }
  return { color: '#ff6a00', fillColor: '#ff9a3d', fillOpacity: 0.62, weight: 1 }
}

function hotspotRadius(h) {
  const b = Number(h.brightness) || 0
  if (b >= 360) return 6.5
  if (b >= 330) return 5.5
  return 4.5
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
        const res = await axios.get(`${baseUrl}/hotspots`, { timeout: 120000 })
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
    <div className="glass-panel p-4 sm:p-6 space-y-4">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="font-semibold text-[14px] sm:text-[15px] text-slate-100 flex items-center gap-2">
            🛰️ Worldwide Live Map — Real Satellite + AI Cameras
          </h2>
          <p className="text-[11px] text-slate-400 font-normal mt-0.5">
            Model is location-agnostic: upload any forest/satellite image from any country — works worldwide. Cameras below are real TFLite predictions. Orange dots are genuine NASA FIRMS satellite detections (past 24h).
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowGlobal((v) => !v)}
          title="Toggle genuine NASA FIRMS satellite hotspots (never simulated — unavailable if NASA unreachable)"
          className={`shrink-0 px-3 py-1.5 rounded-xl font-semibold text-[11px] sm:text-xs transition-all active:scale-[0.98] ${
            showGlobal ? 'btn-primary' : 'btn-ghost'
          }`}
        >
          🛰️ {showGlobal ? 'Hide global fires' : 'Show global fires'}
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
        className="rounded-xl border border-white/10"
        style={{ height: '410px', width: '100%' }}
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Streets (OSM)">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> | Hotspots: <a href="https://firms.modaps.eosdis.nasa.gov/">NASA FIRMS</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              noWrap={false}
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Satellite (Esri)">
            <TileLayer
              attribution='Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics | Hotspots: <a href="https://firms.modaps.eosdis.nasa.gov/">NASA FIRMS</a>'
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              noWrap={false}
            />
          </LayersControl.BaseLayer>
        </LayersControl>
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
                      <p className="text-xs">Real TFLite confidence: {pct}%</p>
                      <p className="text-xs opacity-70">{formatTime(status.timestamp)}</p>
                      <p className="text-[10px] opacity-60">Model inferenced live — works on any forest image worldwide (location-agnostic MobileNetV2)</p>
                    </div>
                  ) : (
                    <p className="text-xs mt-1 opacity-70">
                      No predictions yet — run Live Monitoring or upload an image from any country (model is worldwide).
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
              radius={hotspotRadius(h)}
              pathOptions={hotspotStyle(h)}
            >
              <Popup>
                <div className="text-sm leading-snug">
                  <p className="font-bold flex items-center gap-1">🔥 Active fire — satellite detected</p>
                  <p className="opacity-70 text-xs">
                    {h.lat.toFixed(3)}, {h.lng.toFixed(3)} • {(h.satellite || 'VIIRS') + (h.daynight ? ` • ${h.daynight}` : '')}
                  </p>
                  <p className="text-xs mt-1">
                    {h.acq_date || ''} {h.acq_time ? `${String(h.acq_time).padStart(4, '0').replace(/(\d{2})(\d{2})/, '$1:$2')} UTC` : ''} • brightness {h.brightness}K
                    {h.frp ? ` • FRP ${h.frp}MW` : ''}
                  </p>
                  {h.confidence && <p className="text-xs opacity-70">VIIRS confidence: {h.confidence}</p>}
                  <p className="text-[11px] opacity-60 mt-1">Genuine NASA FIRMS • not a model prediction • <a className="underline" href="https://firms.modaps.eosdis.nasa.gov/" target="_blank" rel="noreferrer">firms.modaps.eosdis.nasa.gov</a></p>
                </div>
              </Popup>
            </CircleMarker>
          ))}
      </MapContainer>

      <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-[11px] sm:text-xs">
        <span className="inline-flex items-center gap-1 text-slate-300"><span className="h-2 w-2 rounded-full bg-[#dc2626] border border-white/60" /> Critical (model)</span>
        <span className="inline-flex items-center gap-1 text-slate-300"><span className="h-2 w-2 rounded-full bg-[#f97316] border border-white/60" /> High (model)</span>
        <span className="inline-flex items-center gap-1 text-slate-300"><span className="h-2 w-2 rounded-full bg-[#eab308] border border-white/60" /> Low (model)</span>
        <span className="inline-flex items-center gap-1 text-slate-300"><span className="h-2 w-2 rounded-full bg-[#22c55e] border border-white/60" /> Safe (model)</span>
        <span className="inline-flex items-center gap-1 text-slate-400"><span className="h-2 w-2 rounded-full bg-[#64748b] border border-white/60" /> No data</span>
        {showGlobal && <span className="inline-flex items-center gap-1 text-orange-300"><span className="h-2 w-2 rounded-full bg-[#ff6a00] border border-white/60" /> NASA satellite</span>}
      </div>

      {showGlobal && hsState === 'loading' && (
        <p className="text-center text-[11px] sm:text-xs text-slate-400">
          🛰️ Fetching genuine worldwide data from NASA FIRMS (VIIRS past 24h)…
        </p>
      )}
      {showGlobal && hsState === 'ready' && (
        <div className="text-center space-y-1">
          <p className="text-[11px] sm:text-xs text-slate-300">
            <span className="inline-flex items-center gap-1.5"><span className={`h-1.5 w-1.5 rounded-full ${hsMeta?.stale ? 'bg-amber-400' : 'bg-emerald-400'} `} /> Genuine</span> • {hsMeta?.count ?? 0} hotspots live worldwide
            {hsMeta?.total > (hsMeta?.count ?? 0) ? ` (brightest ${(hsMeta?.count ?? 0)} of ${hsMeta.total})` : ''} • {hsMeta?.source ?? 'NASA FIRMS'}
            {hsMeta?.fetched_at ? ` • fetched ${new Date(hsMeta.fetched_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} UTC` : ''}
          </p>
          {hsMeta?.stale && <p className="text-[11px] text-amber-300/90">Cached while NASA refreshes — stale-while-revalidate (never fake)</p>}
        </div>
      )}
      {showGlobal && hsState === 'unavailable' && (
        <p className="text-center text-[11px] sm:text-xs text-slate-500">
          🛰️ Live satellite layer unavailable — server has no <code className="px-1 py-0.5 rounded bg-white/10">FIRMS_MAP_KEY</code>. Nothing simulated — set the key and redeploy to see real global fires.
        </p>
      )}

      <LastUpdated timestamp={latestUpdate} />
    </div>
  )
}
