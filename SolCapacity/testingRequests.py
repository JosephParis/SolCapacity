import os

import requests

__priv_key = "6x7x9bbru17Pts8NmuQ3nfCq3Dq1qgI3cMecGUwI"
def testing():
    #await input('Enter your clash royale id:')
    payload = {'api_key': __priv_key, 'format': 'json', 'lat': 35.45, 'lon': -82.98}
    headers = {"Authorization": f"Bearer {__priv_key}"}
    response = requests.get(f"https://developer.nrel.gov/api/solar/solar_resource/v1.json?{__priv_key}", params=payload, headers=headers)
    data = response.json()
    response.headers.get("X-RateLimit-Remaining")
    result = ''
    result += str(data['outputs']['avg_dni']['annual'])
    print(result)
testing()
