/**
 * Worldwide camera monitoring network.
 * Original 10 SoCal cameras kept unchanged (camera-01..10) for
 * backwards compatibility with persisted history; 14 high-risk
 * forest zones added across every continent so the map covers the
 * entire world population and nature. Edit coordinates / zone names
 * here — everything else (map, live mode, background simulation)
 * reads from this file.
 */
export const CAMERA_LOCATIONS = [
  { id: 'camera-01', name: 'Camera 01', zone: 'Angeles Forest - Mt. Wilson', lat: 34.2244, lng: -118.0682 },
  { id: 'camera-02', name: 'Camera 02', zone: 'Malibu Hills - Topanga', lat: 34.1069, lng: -118.6017 },
  { id: 'camera-03', name: 'Camera 03', zone: 'San Gabriel Foothills - Altadena', lat: 34.1897, lng: -118.1312 },
  { id: 'camera-04', name: 'Camera 04', zone: 'Santa Monica Mtns - Malibu Creek', lat: 34.1000, lng: -118.7500 },
  { id: 'camera-05', name: 'Camera 05', zone: 'Cajon Pass - San Bernardino Edge', lat: 34.3076, lng: -117.4693 },
  { id: 'camera-06', name: 'Camera 06', zone: 'Griffith Park Hills', lat: 34.1367, lng: -118.2942 },
  { id: 'camera-07', name: 'Camera 07', zone: 'Placerita Canyon - Santa Clarita', lat: 34.3800, lng: -118.4600 },
  { id: 'camera-08', name: 'Camera 08', zone: 'Point Mugu - Western Malibu', lat: 34.0800, lng: -119.0200 },
  { id: 'camera-09', name: 'Camera 09', zone: 'Big Bear Foothills', lat: 34.2439, lng: -116.9114 },
  { id: 'camera-10', name: 'Camera 10', zone: 'Rolling Hills - Palos Verdes', lat: 33.7444, lng: -118.3456 },
  { id: 'camera-11', name: 'Camera 11', zone: 'Amazon - Manaus, Brazil', lat: -3.1190, lng: -60.0217 },
  { id: 'camera-12', name: 'Camera 12', zone: 'Congo Basin - Kisangani, DRC', lat: 0.5153, lng: 25.1911 },
  { id: 'camera-13', name: 'Camera 13', zone: 'Siberia - Yakutsk, Russia', lat: 62.0355, lng: 129.6755 },
  { id: 'camera-14', name: 'Camera 14', zone: 'Victoria Bush - Melbourne, Australia', lat: -37.8136, lng: 146.3484 },
  { id: 'camera-15', name: 'Camera 15', zone: 'Boreal - Alberta, Canada', lat: 53.5444, lng: -113.4909 },
  { id: 'camera-16', name: 'Camera 16', zone: 'Mediterranean - Athens, Greece', lat: 38.0272, lng: 23.7166 },
  { id: 'camera-17', name: 'Camera 17', zone: 'Iberia - Valencia, Spain', lat: 39.4699, lng: -0.3763 },
  { id: 'camera-18', name: 'Camera 18', zone: 'Andes - Santiago, Chile', lat: -33.4489, lng: -70.6693 },
  { id: 'camera-19', name: 'Camera 19', zone: 'Fynbos - Cape Town, South Africa', lat: -33.9249, lng: 18.4241 },
  { id: 'camera-20', name: 'Camera 20', zone: 'Borneo - Palangkaraya, Indonesia', lat: -2.2091, lng: 113.9169 },
  { id: 'camera-21', name: 'Camera 21', zone: 'Taiga - Alaska, USA', lat: 64.8378, lng: -147.7164 },
  { id: 'camera-22', name: 'Camera 22', zone: 'Anatolia - Antalya, Turkey', lat: 36.8841, lng: 30.7056 },
  { id: 'camera-23', name: 'Camera 23', zone: 'Western Ghats - Kerala, India', lat: 10.8505, lng: 76.2711 },
  { id: 'camera-24', name: 'Camera 24', zone: 'Sichuan Forest - Chengdu, China', lat: 30.5728, lng: 104.0668 },
]

export function getCameraById(id) {
  return CAMERA_LOCATIONS.find((c) => c.id === id) ?? null
}
