import json
import time
from pathlib import Path
import pandas as pd
import requests

BASE = "https://www.wagework.go.kr"
OUT = Path(__file__).resolve().parent
CATEGORY_PAGE = f"{BASE}/pt/b/a/retrieveCtgrSrch.do?topPageId=PT06000000&pageId=PT06010200"
LIST_API = f"{BASE}/pt/b/a/retrieveCatgSrchListData.do"
DETAIL_PAGE = f"{BASE}/pt/b/a/retrieveOccpNvgtDtal.do"
DETAIL_API = f"{BASE}/pt/b/a/retrieveOccpNvgtDtalData.do"
SUMMARY_API = f"{BASE}/pt/b/a/retrieveOccpNvgtSumrData.do"
ABILITY_API = f"{BASE}/pt/b/a/retrieveOccpNvgtChtrAbltData.do"

LIST_OUT = OUT / "임금직업포털_전체직업목록.csv"
DETAIL_OUT = OUT / "임금직업포털_직업상세.csv"
ABILITY_OUT = OUT / "임금직업포털_업무수행능력_긴형식.csv"
FAIL_OUT = OUT / "임금직업포털_수집실패.csv"

CATEGORIES = [str(i) for i in range(10)]
DELAY = 0.25

def session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "ajax-forward": "ajax",
        "Origin": BASE,
    })
    s.get(CATEGORY_PAGE, timeout=60).raise_for_status()
    return s

def post_json(s, url, data, referer):
    for i in range(4):
        try:
            r = s.post(url, data=data, headers={"Referer": referer}, timeout=60)
            r.raise_for_status()
            j = r.json()
            if not j.get("success"):
                raise RuntimeError(j.get("message", "API error"))
            return j
        except Exception:
            if i == 3:
                raise
            time.sleep(1.5 * (i + 1))

def collect_jobs(s):
    rows = []
    for cat in CATEGORIES:
        first = post_json(s, LIST_API, {
            "occpClCd": cat,
            "currentPageNo": 1,
            "recordCountPerPage": 100,
            "orderType": ""
        }, CATEGORY_PAGE)
        info = first["data"]["paginationInfo"]
        pages = int(info["totalPageCount"])
        print(f"[분류 {cat}] {info['totalRecordCount']}개")
        for p in range(1, pages + 1):
            payload = first if p == 1 else post_json(s, LIST_API, {
                "occpClCd": cat,
                "currentPageNo": p,
                "recordCountPerPage": 100,
                "orderType": ""
            }, CATEGORY_PAGE)
            for row in payload["data"]["list"]:
                row["first_category_code"] = cat
                rows.append(row)
            time.sleep(DELAY)
    df = pd.DataFrame(rows).drop_duplicates("konetOccpCd")
    df.to_csv(LIST_OUT, index=False, encoding="utf-8-sig")
    return df

def to_long(code, name, data):
    out = []
    mapping = {
        "ipcrQtntList": "importance_percentile",
        "ipcrScorList": "importance_score",
        "grdQtntList": "level_percentile",
        "grdScorList": "level_score",
    }
    for key, label in mapping.items():
        for rank, item in enumerate(data.get(key, []), 1):
            out.append({
                "konetOccpCd": code,
                "konetOcnm": name,
                "metric_type": label,
                "rank": rank,
                "ability_name": item.get("qsnrNm", ""),
                "ability_description": item.get("qsnrCn", ""),
                "value": item.get("cnt", ""),
            })
    return out

def main():
    s = session()
    jobs = pd.read_csv(LIST_OUT, dtype=str).fillna("") if LIST_OUT.exists() else collect_jobs(s)

    existing = {}
    if DETAIL_OUT.exists():
        d = pd.read_csv(DETAIL_OUT, dtype=str).fillna("")
        existing = {r["konetOccpCd"]: r.to_dict() for _, r in d.iterrows()}
    detail_rows = list(existing.values())

    long_rows = []
    if ABILITY_OUT.exists():
        long_rows = pd.read_csv(ABILITY_OUT, dtype=str).fillna("").to_dict("records")

    failures = []

    for i, row in jobs.iterrows():
        code = row["konetOccpCd"]
        name = row.get("konetOcnm", "")
        if code in existing and existing[code].get("ability_json"):
            print(f"[{i+1}/{len(jobs)}] 건너뜀: {name}")
            continue

        referer = f"{DETAIL_PAGE}?konetOccpCd={code}&topPageId=PT06000000&pageId=PT06010200"
        try:
            s.get(referer, timeout=60).raise_for_status()
            detail = post_json(s, DETAIL_API, {"konetOccpCd": code}, referer)["data"]
            summary = post_json(s, SUMMARY_API, {"konetOccpCd": code}, referer)["data"]
            ability = post_json(s, ABILITY_API, {"konetOccpCd": code}, referer)["data"]

            rec = {
                "konetOccpCd": code,
                "konetOcnm": name,
                "dtyOtlnCn": row.get("dtyOtlnCn", ""),
                "first_category_code": row.get("first_category_code", ""),
                "avwgCnvAmt": row.get("avwgCnvAmt", ""),
                "avwgAmtNm": row.get("avwgAmtNm", ""),
                "psctCd": row.get("psctCd", ""),
                "psctCdNm": row.get("psctCdNm", ""),
                "detail_url": referer,
                "detail_json": json.dumps(detail, ensure_ascii=False),
                "summary_json": json.dumps(summary, ensure_ascii=False),
                "ability_json": json.dumps(ability, ensure_ascii=False),
            }

            detail_rows = [r for r in detail_rows if r.get("konetOccpCd") != code] + [rec]
            long_rows = [r for r in long_rows if r.get("konetOccpCd") != code] + to_long(code, name, ability)

            pd.DataFrame(detail_rows).to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")
            pd.DataFrame(long_rows).to_csv(ABILITY_OUT, index=False, encoding="utf-8-sig")
            print(f"[{i+1}/{len(jobs)}] 저장: {name}")
            time.sleep(DELAY)

        except Exception as e:
            failures.append({"konetOccpCd": code, "konetOcnm": name, "reason": str(e)})
            pd.DataFrame(failures).to_csv(FAIL_OUT, index=False, encoding="utf-8-sig")
            print(f"[{i+1}/{len(jobs)}] 실패: {name} / {e}")

    print("완료")
    print(LIST_OUT)
    print(DETAIL_OUT)
    print(ABILITY_OUT)

if __name__ == "__main__":
    main()
