from urllib.parse import quote
import hashlib, os, sys
import json, tqdm
import tiktoken
import shutil
import base64
import argparse
from scholarly import scholarly
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import traceback
import numpy as np
import gspread
from urllib.parse import quote

def format_score(s): return np.round(s, 4)

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
        author = ' and '.join([f"{author['name']}" for author in authors])
        item = {'title': entry_data['title'], 'author': author, 'abstract': entry_data['summary'], 'link':entry_data['id']}
        try:
            feat = get_text_embedding(item['title'],item['author'],item['abstract'], args.openai_token)
            item['embedding'] = feat
            item['updated'] = entry_data['updated']
            entries.append(item)
        except:
            print(traceback.print_exc())
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
            print (i,entry['title'], last_updated)
            if last_updated > cutoff_dt: keep_entries.append(entry)
        if last_updated is None or (last_updated < cutoff_dt):
            print (f'last updated: {last_updated}, cutoff: {cutoff_dt}')
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

def get_text_embedding(title, author, abstract, openai_token, cache=None):
    # text = f'title: {title}\nauthors: {author}\nabstract: {abstract}\n'
    text = f'title: {title}\nabstract: {abstract}\n'
    num_tokens = num_tokens_from_string(text, "cl100k_base")
    if num_tokens > 1000: text = text[:1000]
    text_md5 = calculate_md5_hash(text)
    if cache is not None:
        if text_md5 in cache: return cache[text_md5]
    print (f'expected #tokens {num_tokens_from_string(text, "cl100k_base")}')
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_token}"
    }
    data = {
        "input":  text,
        "model": "text-embedding-3-small"
    }
    response = requests.post(url, headers=headers, json=data, timeout=10)
    res = response.json()
    feat = res['data'][0]['embedding']
    if cache is not None:
        cache[text_md5] = feat
    return feat

def get_prefilled_google_form(title,author,abstract,link=""):
    entry_title = 'entry.1380929848'
    entry_author = 'entry.906535625'
    entry_abstract = 'entry.1292438233'
    entry_link = 'entry.1838667208'
    entry_rating = 'entry.124074799'
    entry_comments = 'entry.1108263184'
    title = quote(title)
    author = quote(author)
    abstract = quote(abstract)
    link = quote(link)
    return f'https://docs.google.com/forms/d/e/1FAIpQLSfSfFqShId9ssA7GWYmvv7m_7qsIao4K__1rDj9BurNNxUPYQ/viewform?{entry_title}={title}&{entry_author}={author}&{entry_abstract}={abstract}&{entry_link}={link}&{entry_rating}=Read'

def get_prefilled_email(title, author, abstract, link, addr='eoppp.rm'):
    subject = "[arXrec] " + title
    encoded_subject = quote(subject)
    body = "Title: " + title + "\n" + "Author: " + author + "\n" + "Abstract: " + abstract + "\n" + "Link: " + link
    encoded_body = quote(body)
    return f"mailto:{addr[::-1]}@gmail.com?subject={encoded_subject}&body={encoded_body}"

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--setup', action='store_true', default=False, help="load your google scholar profile to get all your publications into the database")
    args.add_argument('--db', type=str, default="db.json", help="path to local json file to store marked publications w/ text embeddings")
    args.add_argument('--openai_token', type=str, default=None, required=True, help='OpenAI API key')
    args.add_argument('--google_oauth', type=str, default=None, required=True, help='Path to Google OAuth Credentials json file')
    args.add_argument('--google_oauth_user', type=str, default=None, required=False, help='Path to Google OAuth Authorized User json file')
    args.add_argument('--cutoff_day', type=int, default=-1, required=False, help='# of days to keep track of on Arxiv, -1 means being adaptive to the arXiv Announcement Schedule')
    args = args.parse_args()

    cutoff_day = args.cutoff_day
    num_keep = 100
    random_ratio = 0.1

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
            try:
                bib = pub['bib']
                title = bib['title']
                author = bib['author']
                if 'abstract' not in bib: continue
                abstract = bib['abstract']
                link = pub['pub_url']
                if pub['bib']['pub_year'] < 2022: continue
                feat = get_text_embedding(title,author,abstract, args.openai_token)
                entry = {'title':title, 'author':author, 'abstract':abstract, 'link':link, 'embedding':feat, 'rating':'read'}
                all_pubs.append(entry)
            except Exception as e:
                print (traceback.format_exc())
        db['read'] = all_pubs
        json.dump(db, open(args.db, 'w'))
        sys.exit(0)
    else:
        assert (os.path.exists(args.db))

    db = json.load(open(args.db))

    ## update db from google forms
    cached_text_embeddings = db.get('cached_text_embeddings', {})
    if args.google_oauth_user:
        google_oauth_user = json.load(open(args.google_oauth_user))
    else:
        google_oauth_user = None
    gc, authorized_user = gspread.oauth_from_dict(json.load(open(args.google_oauth)), google_oauth_user)
    sheet = gc.open('Daily Paper Reading (Responses)').sheet1
    rows = sheet.get_all_values()
    header = rows[0]
    records = []
    online_db_items = []
    for r in rows[1:]:
        rec = dict([[header[idx],r[idx]] for idx in range(len(header))])
        if rec['Title'] == '': continue
        records.append(rec)
        rating = rec['Rating'].lower()
        embedding = get_text_embedding(rec['Title'],rec['Authors'],rec['Abstract'], args.openai_token, cached_text_embeddings)
        db_item = {'title':rec['Title'], 'author':rec['Authors'], 'abstract':rec['Abstract'], 'link':rec['Link'], 'embedding':embedding, 'rating':rating}
        online_db_items.append(db_item)
    db['cached_text_embeddings'] = cached_text_embeddings
    json.dump(db, open(args.db, 'w'))

    for db_item in online_db_items: db.setdefault(db_item['rating'], []).append(db_item)

    if cutoff_day > 0:
        cutoff_dt = (datetime.now() + timedelta(days=-cutoff_day)).strftime('%Y-%m-%d %H:%M:%S')
    else:
        weekday = datetime.now().weekday()+1
        if weekday in [2,3,4]:
            cutoff_dt = (datetime.now() + timedelta(days=-1)).strftime('%Y-%m-%d') + ' 09:50:00'
        elif weekday in [1,7]:
            cutoff_dt = (datetime.now() + timedelta(days=-3)).strftime('%Y-%m-%d') + ' 09:50:00'
        else:
            print ('no annoucement on Friday per https://info.arxiv.org/help/availability.html')
            sys.exit(0)
    new_entries = fetch_from_date(cutoff_dt)
    json.dump(new_entries, open(f'workspace/new_entries_from_{cutoff_dt.replace(":","").replace("-","")}.json', 'w'))
    if len(new_entries) == 0:
        print ('no new entries')
        sys.exit(0)
    # relevancy = (\sum READ*2*s + \sum like*10*s + \sum IGNORE*-5*s)
    db_feats = []
    db_weights = []
    db_titles = []
    for [w,tag] in [[5,'read'],[2,'like'],[-5,'ignore']]:
        feats = [e['embedding'] for e in db.get(tag,[])]
        db_titles.extend([e for e in db.get(tag,[])])
        db_feats.extend(feats)
        db_weights.extend([w]*len(feats))
    new_feats = np.array([e['embedding'] for e in new_entries])
    new_feats = new_feats/np.linalg.norm(new_feats, axis=1, keepdims=True)
    db_feats = np.array(db_feats)
    db_feats = db_feats/np.linalg.norm(db_feats, axis=1, keepdims=True)
    db_weights = np.array(db_weights)
    batch_size = 32
    topK = 3
    for i in tqdm.tqdm(range(0, len(new_entries), batch_size)):
        batch_feats = new_feats[i:i+batch_size]
        batch_scores = np.dot(batch_feats, db_feats.T)
        batch_nn = np.argsort(-batch_scores, axis=1)[:,:topK]
        for j in range(batch_scores.shape[0]):
            relevancy = 0
            for k in range(topK):
                ind = batch_nn[j,k]
                relevancy += (batch_scores[j,ind]*db_weights[ind])
            relevancy = relevancy/topK
            new_entries[i+j]['relevancy'] = format_score(relevancy)
            new_entries[i+j]['topK'] = []
            for k in batch_nn[j,:topK].tolist():
                new_entries[i+j]['topK'].append(
                    {'title': db_titles[k]['title'],'link': db_titles[k]['link'],
                     'similarity': format_score(float(batch_scores[j,k]))})
    new_entries = sorted(new_entries, key=lambda x: x['relevancy'], reverse=True)

    num_keep_top = int(num_keep*(1-random_ratio))
    selected_entries = new_entries[:num_keep_top]
    if len(new_entries) > num_keep_top:
        selected_entries.extend(np.random.choice(new_entries[num_keep_top:], size=int(num_keep*random_ratio), replace=False))

    def msg2html(e):
        return json.dumps({'title': e['title'], 'author': e['author'],
                'abstract': e['abstract'], 'link': e['link'],
                'date': e['updated'].split('T')[0],
                'relevancy': e['relevancy'],
               'topK': e['topK'],
               'mailto': e['mailto'],
                'form': get_prefilled_google_form(e['title'], e['author'], e['abstract'], e['link'])}) + ","

    for e in selected_entries:
        e['mailto'] = get_prefilled_email(e['title'], e['author'], e['abstract'], e['link'])

    page_fpath = 'docs/index.html'
    lines = [l for l in open('_page.html').readlines()]
    content_line_ind = [l.strip().rstrip() for l in lines].index('__CONTENT__')
    lines[content_line_ind] = '\n'.join(msg2html(e) for e in selected_entries) + '\n'
    minified = ''.join(lines) + '\n'
    with open(page_fpath, 'w') as f:
        f.write(minified)
    os.system(f'git add {page_fpath}')
    # keep a daily copy
    datestr = datetime.now().strftime('%Y%m%d')
    daily_path_fpath = f'docs/daily{datestr}.html'
    shutil.copy(page_fpath, daily_path_fpath)
    os.system(f'git add {daily_path_fpath}')
    os.system(f'git commit -m "updated at {cutoff_dt}"')
    os.system(f'git push')
