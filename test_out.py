import requests

stats = requests.get('http://127.0.0.1:5000/stats').json()
print('=== STATS ===')
print(stats)

hot = requests.get('http://127.0.0.1:5000/hotspots').json()
print('\n=== HOTSPOTS ===')
print(hot)

config = requests.get('http://127.0.0.1:5000/configure-alert-email').json()
print('\n=== ALERTS (no email) ===')
print(config)
