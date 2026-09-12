# Sample images for Live Monitoring mode

Place exactly these 5 files in this folder (`frontend/public/samples/`):

1. `sample-1.jpg`
2. `sample-2.jpg`
3. `sample-3.jpg`
4. `sample-4.jpg`
5. `sample-5.jpg`

The app references them at these URLs (Vite serves `public/` at the site root):

- `/samples/sample-1.jpg` → Camera 01
- `/samples/sample-2.jpg` → Camera 02
- `/samples/sample-3.jpg` → Camera 03
- `/samples/sample-4.jpg` → Camera 04
- `/samples/sample-5.jpg` → Camera 05

Tips:
- Use real JPG forest/satellite photos (the same kind you upload manually).
- For a good demo mix, make ~2 images show fire/smoke and ~3 show healthy forest.
- Any `.jpg`/`.jpeg` content works — just keep these exact filenames.
  (If you only have `.png` files, either convert them to JPG or update the
  `SAMPLE_IMAGES` array in `frontend/src/components/LiveMonitor.jsx`.)
- No dev-server restart needed: just drop the files in and refresh the page.
- If a file is missing, Live Monitoring shows a friendly
  "Sample image missing" placeholder and keeps cycling.
