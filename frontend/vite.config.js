import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // optional: lets frontend call /predict relatively if you want
      // '/predict': 'http://localhost:5000',
      // '/health': 'http://localhost:5000',
    },
  },
})
