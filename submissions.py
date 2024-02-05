import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

def query_arxiv(api_url, search_query, start_i=0):
    params = {
        'search_query': search_query,
        'start': start_i,
        'max_results': 50,  # You can adjust the number of results as needed
    }

    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code}")
        return None

def parse_arxiv_response(xml_response):
    root = ET.fromstring(xml_response)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = []

    for entry in root.findall('.//atom:entry',ns):
        entry_data = {}
        for child in entry:
            tag = child.tag.replace(f"{{{ns['atom']}}}", "")
            if tag == 'author': continue
            entry_data[tag] = child.text
        authors = []
        for author_elem in entry.findall('.//atom:author', ns):
            author_info = {}
            for author_child in author_elem:
                author_tag = author_child.tag.replace(f"{{{ns['atom']}}}", "")
                if author_tag == 'name':
                    author_info['name'] = author_child.text
                elif author_tag == 'affiliation':
                    author_info['affiliation'] = author_child.text
            authors.append(author_info)
        entry_data['author'] = authors
        entries.append(entry_data)
    return entries

def fetch_from_date(cutoff_date): # YYYY-MM-DD

    cutoff_dt = datetime.strptime(cutoff_date, '%Y-%m-%d')

    api_url = 'http://export.arxiv.org/api/query?sortBy=lastUpdatedDate&sortOrder=descending'
    search_query = 'cat:cs.AI OR cat:cs.CV OR cat:cs.LG OR cat:cs.RO'

    start_i = 0
    keep_entries = []
    while True:
        xml_response = query_arxiv(api_url, search_query, start_i=start_i)
        if not xml_response: break
        entries = parse_arxiv_response(xml_response)
        last_updated = None
        for i, entry in enumerate(entries):
            last_updated = datetime.strptime(entry['updated'], '%Y-%m-%dT%H:%M:%SZ')
            if last_updated > cutoff_dt: keep_entries.append(entry)
        if last_updated is None or (last_updated < cutoff_dt):
            break
        start_i += len(entries)
    return keep_entries

if __name__ == "__main__":
    fetch_from_date('2024-02-01')

