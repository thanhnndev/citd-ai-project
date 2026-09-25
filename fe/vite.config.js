import { defineConfig } from 'vite'

// The frontend is intentionally self-contained. There is no /api proxy: all
// demo responses are generated from src/demo-data.js in the browser bundle.
export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
})
