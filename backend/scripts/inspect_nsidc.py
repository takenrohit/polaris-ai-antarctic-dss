import urllib.request
import re

url = 'https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
years = re.findall(r'href="(\d{4})/', html)
print('NSIDC available years:', years[-6:])

# Check files in the latest year (e.g. 2024 or 2025)
latest_year = years[-1]
year_url = f"{url}{latest_year}/"
html_year = urllib.request.urlopen(urllib.request.Request(year_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=10).read().decode('utf-8')
months = re.findall(r'href="(\d{2}_[a-zA-Z]+)/', html_year)
print(f'Months in {latest_year}:', months)

if months:
    month_url = f"{year_url}{months[0]}/"
    html_month = urllib.request.urlopen(urllib.request.Request(month_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=10).read().decode('utf-8')
    tifs = re.findall(r'href="([^"]+\.tif)"', html_month)
    print(f'TIF files in {months[0]}:', len(tifs))
    if tifs:
        print('Sample TIF URL:', month_url + tifs[0])
