import requests
from bs4 import BeautifulSoup
import json
# https://www.youtube.com/watch?v=15f4JhJ8SiQ&ab_channel=JohnWatsonRooney

region_urls = {
    'North America': 'https://avibase.bsc-eoc.org/checklist.jsp?region=NAM',
    'South America': 'https://avibase.bsc-eoc.org/checklist.jsp?region=SAM',
    'Europe': 'https://avibase.bsc-eoc.org/checklist.jsp?region=EUR',
    'Africa': 'https://avibase.bsc-eoc.org/checklist.jsp?region=AFR',
    'Asia': 'https://avibase.bsc-eoc.org/checklist.jsp?region=ASI',
    'Oceania': 'https://avibase.bsc-eoc.org/checklist.jsp?region=OCE',
    'Antarctica': 'https://avibase.bsc-eoc.org/checklist.jsp?region=AQ'
}
data = {region: [] for region in region_urls}
# data = {"region": [], "common_name": [], "species_name": []}
for region, url in region_urls.items():
    html = requests.get(url)

    text = BeautifulSoup(html.text, 'html.parser')

    birds_table = text.find('table', class_ = 'table')

    for row in birds_table.find_all('tr', class_ = 'highlight1'):
        common_name = row.find('td').text
        species_name = row.find('i').text
        data[region].append(species_name.lower())

with open('birds_by_region.json', 'w') as f:
    json.dump(data, f, indent = 4)
