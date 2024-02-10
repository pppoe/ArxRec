from urllib.parse import quote
import hashlib, os, sys
import json, tqdm
import tiktoken
import base64
import argparse
from scholarly import scholarly
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import numpy as np

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

    for entry in tqdm.tqdm(root.findall('.//atom:entry',ns)):
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
        author = ' and '.join([f"{author['name']}" for author in authors])
        item = {'title': entry_data['title'], 'author': author, 'abstract': entry_data['summary'], 'link':entry_data['id']}
        feat = get_text_embedding(item['title'],item['author'],item['abstract'], args.openai_token)
        item['embedding'] = feat
        item['updated'] = entry_data['updated']
        entries.append(item)
    return entries

def fetch_from_date(cutoff_datetime): # YYYY-MM-DD HH:MM:SS
    cutoff_dt = datetime.strptime(cutoff_datetime, '%Y-%m-%d %H:%M:%S')
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

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def calculate_md5_hash(input_string):
    md5_hash = hashlib.md5(input_string.encode()).hexdigest()
    return md5_hash

def get_text_embedding(title, author, abstract, openai_token):
    text = f'title: {title}\nauthors: {author}\nabstract: {abstract}\n'
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_token}"
    }
    data = {
        "input":  text,
        "model": "text-embedding-3-small"
    }
    response = requests.post(url, headers=headers, json=data)
    res = response.json()
    feat = res['data'][0]['embedding']
    return feat

def get_prefilled_google_form(title,author,abstract,link="",embedding=""):
    entry_title = 'entry.1380929848'
    entry_author = 'entry.906535625'
    entry_abstract = 'entry.1292438233'
    entry_link = 'entry.1838667208'
    entry_rating = 'entry.124074799'
    entry_comments = 'entry.1108263184'
    entry_embedding = 'entry.1453077783'
    title = quote(title)
    author = quote(author)
    abstract = quote(abstract)
    link = quote(link)
    embedding = quote(embedding)
    return f'https://docs.google.com/forms/d/e/1FAIpQLSfSfFqShId9ssA7GWYmvv7m_7qsIao4K__1rDj9BurNNxUPYQ/viewform?{entry_title}={title}&{entry_author}={author}&{entry_abstract}={abstract}&{entry_link}={link}&{entry_rating}=Read&{entry_embedding}={embedding}'

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--setup', action='store_true', default=False, help="load your google scholar profile to get all your publications into the database")
    args.add_argument('--db', type=str, default="db.json", help="path to local json file to store marked publications w/ text embeddings")
    args.add_argument('--openai_token', type=str, default=None, required=True, help='OpenAI API key')
    args = args.parse_args()

    cutoff_day = 1
    num_keep = 10
    random_ratio = 0.2

    # structure: { 'read/mark/ignore' : [pub1, pub2, ...] }

    if args.setup:
        db = {}
        search_query = scholarly.search_author('Haoxiang Li')
        # Retrieve the first result from the iterator 
        first_author_result = next(search_query)
        # Retrieve all the details for the author
        author = scholarly.fill(first_author_result)
        # If the first one is not the one you are looking for, keep moving
        all_pubs = []
        for pub in tqdm.tqdm(author['publications']):
            scholarly.fill(pub)
            bib = pub['bib']
            title = bib['title']
            author = bib['author']
            if 'abstract' not in bib: continue
            abstract = bib['abstract']
            link = pub['pub_url']
            feat = get_text_embedding(title,author,abstract, args.openai_token)
            entry = {'title':title, 'author':author, 'abstract':abstract, 'link':link, 'embedding':feat, 'rating':'Read'}
            all_pubs.append(entry)
        db['read'] = all_pubs
        json.dump(db, open(args.db, 'w'))
        sys.exit(0)
    else:
        assert (os.path.exists(args.db))

    db = json.load(open(args.db))
    cutoff_dt = (datetime.now() + timedelta(days=-cutoff_day)).strftime('%Y-%m-%d %H:%M:%S')
    # new_entries = fetch_from_date(cutoff_dt)
    # json.dump(new_entries, open('/tmp/new_entries.json', 'w'))
    new_entries = json.load(open('/tmp/new_entries.json'))
    if len(new_entries) == 0: sys.exit(0)
    # relevancy = (\sum READ*2*s + \sum MARK*1*s + \sum IGNORE*-2*s)
    db_feats = []
    db_weights = []
    for [w,tag] in [[2,'read'],[1,'mark'],[-2,'ignore']]:
        feats = [e['embedding'] for e in db.get(tag,[])]
        db_feats.extend(feats)
        db_weights.extend([w]*len(feats))
    new_feats = np.array([e['embedding'] for e in new_entries])
    new_feats = new_feats/np.linalg.norm(new_feats, axis=1, keepdims=True)
    db_feats = np.array(db_feats)
    db_feats = db_feats/np.linalg.norm(db_feats, axis=1, keepdims=True)
    db_weights = np.array(db_weights).reshape(-1,1)
    batch_size = 32 # limit memory footprint
    for i in tqdm.tqdm(range(0, len(new_entries), batch_size)):
        batch_feats = new_feats[i:i+batch_size]
        batch_scores = np.dot(batch_feats, db_feats.T)
        batch_relevancy = (batch_scores @ db_weights).reshape(-1)/len(db_weights)
        for j, score in enumerate(batch_relevancy):
            new_entries[i+j]['relevancy'] = score
    new_entries = sorted(new_entries, key=lambda x: x['relevancy'], reverse=True)

    num_keep_top = int(num_keep*(1-random_ratio))
    selected_entries = new_entries[:num_keep_top]
    if len(new_entries) > num_keep_top:
        selected_entries.extend(np.random.choice(new_entries[num_keep_top:], size=int(num_keep*random_ratio), replace=False))

