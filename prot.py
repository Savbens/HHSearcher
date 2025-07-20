import time
import requests
import pdfplumber
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip('•—\n\t ')

def extract_resume_sections(pdf_path: str) -> dict[str, str]:
    sections = defaultdict(list)
    current = 'контакты'
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text(
                x_tolerance=3,
                y_tolerance=3,
                layout=True,
                keep_blank_chars=False
            )
            if not raw:
                continue
            lines = [clean_text(line) for line in raw.split('\n') if clean_text(line)]
            for line in lines:
                if "опыт работы" in line.lower():
                    current = "опыт работы"
                elif "образование" in line.lower():
                    current = "образование"
                elif "навыки" in line.lower():
                    current = "навыки"
                elif "дополнительная информация" in line.lower():
                    current = "дополнительная информация"
                if line and not line.isspace():
                    sections[current].append(line)
    return {sec: "\n".join(blocks).strip() for sec, blocks in sections.items()}

resume_sections = extract_resume_sections('resume1.pdf')
resume_text = ""
if "опыт работы" in resume_sections:
    resume_text += "Опыт работы:\n" + resume_sections["опыт работы"] + "\n\n"
if "навыки" in resume_sections:
    resume_text += "Навыки и технологии:\n" + resume_sections["навыки"] + "\n\n"
if "образование" in resume_sections:
    resume_text += "Образование:\n" + resume_sections["образование"] + "\n\n"
if "дополнительная информация" in resume_sections:
    resume_text += resume_sections["дополнительная информация"] + "\n\n"
if "контакты" in resume_sections:
    resume_text += resume_sections["контакты"]
print(f"[+] Из PDF извлечено и отформатировано {len(resume_text)} символов текста.")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
profile_emb = model.encode(resume_text, convert_to_tensor=True, device='cpu')
print(resume_text)
KEYWORDS = ["machine learning", "data science", "C++ developer"]
LOCATIONS = [
    {"area": 2,     "name": "Санкт-Петербург"},
    {"remote": True,"name": "Удалённая"}
]
FILTER_TERMS = ["machine learning", "data science", "С++"]
session = requests.Session()
session.headers.update({
    'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/112.0.0.0 Safari/537.36',
    'Accept-Language': 'ru-RU,ru;q=0.9',
    'Referer':         'https://hh.ru/'
})

def search_vacancies_html(query: str, loc: dict) -> list[dict]:
    url = 'https://hh.ru/search/vacancy'
    params = {'text': query, 'items_on_page': 20}
    if 'area' in loc:
        params['area'] = loc['area']
    if loc.get('remote'):
        params['remote'] = 1
    r = session.get(url, params=params)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    out = {}
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/vacancy/' not in href:
            continue
        name = a.get_text(strip=True)
        if not name:
            continue
        snippet = ""
        sib = a.find_next_sibling()
        if sib and sib.name == 'div':
            snippet = sib.get_text(" ", strip=True)
        out[href] = {
            'name':     name,
            'url':      href,
            'snippet':  snippet,
            'location': loc['name']
        }
    return list(out.values())

all_vac = []
for loc in LOCATIONS:
    for kw in KEYWORDS:
        items = search_vacancies_html(kw, loc)
        all_vac.extend(items)
        time.sleep(0.5)
unique = {v['url']: v for v in all_vac}.values()
filtered = []
for v in unique:
    text = (v['name'] + " " + v['snippet']).lower()
    if any(term in text for term in FILTER_TERMS):
        filtered.append(v)
print(f"[+] Всего после лексической фильтрации: {len(filtered)} вакансий")
if not filtered:
    raise SystemExit("Нет ни одной вакансии после фильтрации по ключевым терминам.")

def get_full_vacancy_text(url: str) -> str:
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        desc = soup.find('div', {'data-qa': 'vacancy-description'})
        if desc:
            return desc.get_text(" ", strip=True)
    except Exception as e:
        print(f"[!] Ошибка при загрузке описания вакансии {url}: {e}")
    return ""

vac_texts = []
for v in filtered:
    full_text = get_full_vacancy_text(v['url'])
    combined_text = f"{v['name']} {v['snippet']} {full_text}".strip()
    vac_texts.append(combined_text)
    time.sleep(0.5)
vac_embs = model.encode(vac_texts, convert_to_tensor=True, device='cpu')
scores = util.cos_sim(profile_emb, vac_embs)[0]
top5_idx = scores.topk(k=20).indices
print("\n🔥 Топ-5 релевантных вакансий:")
for idx in top5_idx:
    v = filtered[idx]
    print(f"- {v['name']} ({v['location']})")
    print(f"    {v['url']}")
    print(f"    Схожесть: {scores[idx]:.2f}\n")
