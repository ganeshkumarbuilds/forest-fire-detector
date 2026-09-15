import requests, re

r = requests.get('https://frontend-fire-detector.vercel.app/', timeout=20)
print('STATUS:', r.status_code)
print('HTML LEN:', len(r.text))
print('--- FIRST 2000 CHARS ---')
print(r.text[:2000])
matches = re.findall(r'https?://[^\"\'`<>\s]+', r.text)
for m in matches[:50]:
    if any(x in m.lower() for x in ['api', '5000', 'render', 'vercel', 'backend', 'predict', 'health']):
        print('FOUND:', m)