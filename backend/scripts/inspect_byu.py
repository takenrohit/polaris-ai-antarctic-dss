import urllib.request
import re

url = 'https://www.scp.byu.edu/data/iceberg/database1.html'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
links = re.findall(r'href="([^"]+)"', html)
print('Total links:', len(links))
for l in links:
    if any(k in l.lower() for k in ['a23', 'a76', 'd28', 'b15', 'table', 'data', 'csv']):
        print('Candidate link:', l)
