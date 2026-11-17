"""Tìm ứng viên ảnh trên Wikimedia Commons (giấy phép mở) cho bộ dữ liệu Day 08.

Chỉ dùng cho giảng viên khi cần dựng lại / mở rộng bộ ảnh. Học viên KHÔNG cần chạy file này.
Dùng: python data/fetch_candidates.py --out work/cands
"""
import argparse, csv, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "ai20k-lab-dataset-builder/1.0 (educational; contact: course-staff)"}
OK_LIC = re.compile(r"^(CC BY(-SA)? [0-9.]+|CC0.*|Public domain.*|PD.*)$", re.I)
QUERIES = [
    "Hanoi street motorbikes", "Ho Chi Minh City traffic motorbikes", "Saigon street rush hour",
    "cyclo Vietnam street", "xe ba gác Vietnam", "tricycle cargo Vietnam street", "Hanoi Old Quarter street",
    "Vietnam street night motorbikes", "Vietnam market street vendors", "tuk-tuk Bangkok street",
    "Phnom Penh street traffic", "Bangkok street night traffic", "Vietnam street rain motorbike",
    "Vietnam bus street", "Vietnam truck street", "Hanoi crosswalk pedestrians", "Da Nang street traffic",
    "Hue street cyclo", "Vientiane street", "Jakarta street motorbikes ojek", "Manila jeepney street",
]

def get(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/cands")
    ap.add_argument("--per-query", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for q in QUERIES:
        try:
            d = get({"action": "query", "generator": "search", "gsrsearch": q + " filetype:bitmap",
                     "gsrnamespace": 6, "gsrlimit": a.per_query, "prop": "imageinfo",
                     "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1024})
        except Exception as e:
            print("skip", q, e, file=sys.stderr); continue
        for p in (d.get("query", {}).get("pages", {})).values():
            ii = (p.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata", {})
            lic = md.get("LicenseShortName", {}).get("value", "")
            if p["title"] in seen or not OK_LIC.match(lic) or ii.get("mime") != "image/jpeg":
                continue
            if ii.get("width", 0) < 1000 or ii.get("height", 0) < 600:
                continue
            seen.add(p["title"])
            author = re.sub(r"<[^>]+>", "", md.get("Artist", {}).get("value", "")).strip()
            rows.append({"title": p["title"].replace("File:", ""), "author": author, "license": lic,
                         "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"), "q": q})
        time.sleep(0.5)
    (out / "cands.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(rows), "ứng viên ->", out / "cands.json")

if __name__ == "__main__":
    main()
