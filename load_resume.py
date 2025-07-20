import re
import pdfplumber
from collections import defaultdict

SECTION_HEADERS = [
    r'(О\s*себе|About|Summary)',
    r'(Опыт\s*работы.*?(?:\d+\s*(?:год|лет|месяц|месяцев))?|Experience.*)',
    r'(Образование|Education)',
    r'(Навыки|Skills)',
    r'(Сертификаты?|Certificates?)',
    r'(Проекты?|Projects?)',
    r'(Дополнительная\s*информация|Additional\s*Info)',
    r'(Контакты?|Contacts?)'
]
HEADER_REGEX = re.compile(r'(' + r'|'.join(SECTION_HEADERS) + r')', re.IGNORECASE)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip('•—\n\t ')

def extract_resume_sections(pdf_path: str, debug: bool = False) -> dict[str, str]:
    sections = defaultdict(list)
    current = 'Контакты'
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
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
                m = HEADER_REGEX.search(line)
                if m:
                    header = m.group(1).lower()
                    if debug:
                        print(f"[DEBUG] Found header: '{header}' in line: '{line}'")
                    if "опыт" in header or "experience" in header:
                        current = "опыт работы"
                    elif "образование" in header or "education" in header:
                        current = "образование"
                    elif "навыки" in header or "skills" in header:
                        current = "навыки"
                    else:
                        current = header
                    remaining_text = re.sub(HEADER_REGEX, '', line).strip()
                    if remaining_text:
                        sections[current].append(remaining_text)
                    continue
                if line and not line.isspace():
                    sections[current].append(line)
    return {sec: "\n".join(blocks).strip() for sec, blocks in sections.items()}

resume_sections = extract_resume_sections('resume.pdf')
for sec, content in resume_sections.items():
    print(f"\n=== {sec.upper()} ({len(content)} символов) ===")
    print(content[:500].rstrip(), "..." if len(content) > 500 else "")
