import requests, re

r = requests.get('https://frontend-fire-detector.vercel.app/assets/index-B5gPVU5C.js', timeout=20)
# Search for the API URL usage pattern
matches = re.findall(r'(https?://[^\"\'`<>\s]{10,80})', r.text)
for m in matches:
    if any(x in m.lower() for x in ['api', 'render', 'localhost', 'vercel', '5000', 'predict', 'health', 'ready']):
        print('FOUND:', m)