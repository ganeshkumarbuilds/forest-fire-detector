import requests, re

r = requests.get('https://frontend-fire-detector.vercel.app/assets/index-B5gPVU5C.js', timeout=20)
for i, line in enumerate(r.text.split('\n')):
    if any(x in line for x in ['VITE_API_URL', 'localhost', 'waitForBackend', 'apiUrl', 'API_URL', '/ready', 'import.meta.env']):
        print(f'{i}: {line[:300]}')