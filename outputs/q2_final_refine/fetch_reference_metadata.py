"""Fetch original-paper metadata only; never load experimental data/models."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,gzip
from urllib.request import Request, urlopen
from html.parser import HTMLParser

class CitationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields={}
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        key=values.get('name','')
        if tag=='meta' and key.startswith('citation_'):
            self.fields.setdefault(key,[]).append(values.get('content',''))

O=Path(__file__).resolve().parent
(O/'qa').mkdir(parents=True,exist_ok=True)
SOURCES={
 'TFN':'https://aclanthology.org/D17-1115/',
 'LMF':'https://aclanthology.org/P18-1209/',
 'MFN':'https://ojs.aaai.org/index.php/AAAI/article/view/12021',
 'MulT':'https://aclanthology.org/P19-1656/',
 'MISA':'https://api.crossref.org/works/10.1145/3394171.3413678',
 'Self-MM':'https://ojs.aaai.org/index.php/AAAI/article/view/17289',
 'MMIM':'https://aclanthology.org/2021.emnlp-main.723/',
 'TFR-Net':'https://api.crossref.org/works/10.1145/3474085.3475585',
 'MissModal':'https://aclanthology.org/2023.tacl-1.94/',
 'M3S':'https://aclanthology.org/2022.aacl-main.10/',
 'MMIN':'https://aclanthology.org/2021.acl-long.203/',
}
def get(pair):
    name,url=pair
    request=Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urlopen(request,timeout=35) as response: raw=response.read()
    if raw[:2]==b'\x1f\x8b': raw=gzip.decompress(raw)
    text=raw.decode('utf-8')
    if 'api.crossref.org' in url:
        m=json.loads(text)['message']
        result={'title':m['title'][0],
                'subtitle':m.get('subtitle',[]),
                'authors':[a.get('given','')+' '+a.get('family','') for a in m['author']],
                'venue':m['container-title'][0], 'year':m['published']['date-parts'][0][0],
                'pages':m.get('page'),'doi':m['DOI'],'url':m['URL'],
                'evidence_note':'Publisher-deposited DOI metadata; official publisher URL retained.'}
    else:
        parser=CitationParser()
        parser.feed(text)
        result={'citation_metadata':parser.fields}
        (O/'qa'/f'{name}_official_metadata.html').write_text(text,encoding='utf-8')
    return name,{'retrieval_url':url,**result}
with ThreadPoolExecutor(max_workers=5) as pool: data=dict(pool.map(get,SOURCES.items()))
(O/'qa/reference_metadata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved metadata for '+str(len(data))+' original papers.')
