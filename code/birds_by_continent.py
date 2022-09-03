import requests
from bs4 import BeautifulSoup
import csv
# https://www.youtube.com/watch?v=15f4JhJ8SiQ&ab_channel=JohnWatsonRooney

urls = {
    'North America': 'https://avibase.bsc-eoc.org/checklist.jsp?region=NAM',
    'South America': 'https://avibase.bsc-eoc.org/checklist.jsp?region=SAM',
    'Europe': 'https://avibase.bsc-eoc.org/checklist.jsp?region=EUR',
    'Africa': 'https://avibase.bsc-eoc.org/checklist.jsp?region=AFR',
    'Asia': 'https://avibase.bsc-eoc.org/checklist.jsp?region=ASI',
    'Oceana': 'https://avibase.bsc-eoc.org/checklist.jsp?region=OCE',
    'Antarctica': 'https://avibase.bsc-eoc.org/checklist.jsp?region=AQ'
}

data = {"continent": [], "common_name": [], "species_name": []}
for continent, url in urls.items():
    html = requests.get(url)

    text = BeautifulSoup(html.text, 'html.parser')

    birds_table = text.find('table', class_ = 'table')

    for row in birds_table.find_all('tr', class_ = 'highlight1'):
        common_name = row.find('td').text
        species_name = row.find('i').text
        data["continent"].append(continent)
        data["common_name"].append(common_name)
        data["species_name"].append(species_name)

with open("birds_by_continent.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(data.keys())
    writer.writerows(zip(*data.values()))
