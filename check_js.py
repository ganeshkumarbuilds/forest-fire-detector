import requests, re

r = requests.get('https://frontend-fire-detector.vercel.app/assets/index-B5gPVU5C.js', timeout=20)
print('JS LEN:', len(r.text))
matches = re.findall(r'https?://[^\"\'`<>\s]+', r.text)
for m in matches:
    if any(x in m.lower() for x in ['api', '5000', 'render', 'vercel', 'backend', 'predict', 'health', 'localhost']):
        print('FOUND:', m)