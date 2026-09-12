import { useMemo } from 'react'
import L from 'leaflet'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import { CAMERA_LOCATIONS } from '../cameraLocations.js'
import { getRiskLevel, getRiskBadgeClass } from './ResultCard.jsx'

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
 * Only live-monitoring results feed it — manual uploads carry no camera.
 */
export default function MapView({ cameraStatus }) {
  const bounds = useMemo(
    () => L.latLngBounds(CAMERA_LOCATIONS.map((c) => [c.lat, c.lng])),
    []
  )

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 p-4 sm:p-6 space-y-4 transition-all duration-300">
      <h2 className="font-bold text-sm sm:text-base text-slate-100">
        🗺️ Monitoring Network Map
      </h2>

      <MapContainer
        bounds={bounds}
        scrollWheelZoom={false}
        className="rounded-xl border border-slate-700"
        style={{ height: '320px', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
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
      </MapContainer>

      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-[11px] sm:text-xs text-slate-400">
        <span>🔴 Critical</span>
        <span>🟠 High risk</span>
        <span>🟡 Low risk</span>
        <span>🟢 Safe</span>
        <span>⚪ No data</span>
      </div>
    </div>
  )
}
