import axios from 'axios'

/**
 * Poll GET /ready until the backend's model finishes loading (cold boot
 * on free hosting sleeps + loads ~500MB TF, which takes a while).
 *
 * Resolves true once ready, false after `timeoutMs`. Never throws.
 * `onTick(elapsedSec)` fires roughly every second for progress UI.
 */
export default async function waitForBackend(apiUrl, { timeoutMs = 150000, onTick = null } = {}) {
  const start = Date.now()
  for (;;) {
    try {
      const res = await axios.get(`${apiUrl}/ready`, { timeout: 10000 })
      if (res.data?.ready) return true
    } catch {
      // Server asleep / booting / unreachable — keep polling.
    }
    const elapsed = Date.now() - start
    if (elapsed >= timeoutMs) return false
    try {
      onTick?.(Math.floor(elapsed / 1000))
    } catch {
      // ignore UI callback errors
    }
    await new Promise((r) => setTimeout(r, 4000))
  }
}
