/**
 * Fixed mock locations for the simulated camera monitoring network.
 * Edit coordinates / zone names here — everything else (map, live mode)
 * reads from this file.
 */
export const CAMERA_LOCATIONS = [
  { id: 'camera-01', name: 'Camera 01', zone: 'Zone A - Ridge North', lat: 34.0522, lng: -118.2437 },
  { id: 'camera-02', name: 'Camera 02', zone: 'Zone B - Valley East', lat: 34.0622, lng: -118.2537 },
  { id: 'camera-03', name: 'Camera 03', zone: 'Zone C - Slope South', lat: 34.0422, lng: -118.2337 },
  { id: 'camera-04', name: 'Camera 04', zone: 'Zone D - Creek West', lat: 34.0722, lng: -118.2637 },
  { id: 'camera-05', name: 'Camera 05', zone: 'Zone E - Summit', lat: 34.0322, lng: -118.2237 },
]

export function getCameraById(id) {
  return CAMERA_LOCATIONS.find((c) => c.id === id) ?? null
}
