import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE = Path(__file__).resolve().parent
DETAIL_FILE = BASE / "임금직업포털_직업상세.csv"
ABILITY_FILE = BASE / "임금직업포털_업무수행능력_긴형식.csv"
OUTPUT_FILE = BASE / "GPT5_직업별_AI시간절감점수.csv"
ERROR_FILE = BASE / "GPT5_직업별_AI시간절감점수_오류.csv"

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TOP_N_IMPORTANCE = 6
TOP_N_LEVEL = 6
REQUEST_INTERVAL_SECONDS = 25
MAX_RETRIES = 6

SYSTEM_PROMPT = '''당신은 직업과 생성형 AI의 기능을 잘 아는 전문가입니다.
주어진 직업에서 현재의 생성형 AI를 보조 도구로 활용할 때 전체 작업시간이 평균적으로 얼마나 줄어드는지 평가하세요.

기준:
- 0.00: AI로 작업시간을 거의 줄일 수 없음
- 1.00: 직업의 전체 업무를 AI가 사실상 전적으로 수행 가능
- 문서작성, 요약, 정보검색, 분석, 번역, 코딩은 시간절감 가능성이 높음
- 현장 이동, 육체노동, 장비 조작, 대면 돌봄, 최종 법적·윤리적 책임은 낮음
- 직업 전체 업무시간을 기준으로 판단
- 설명 없이 0.00~1.00의 숫자 하나만 출력
'''

def parse_score(text):
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text.strip())
    if not m:
        raise ValueError(f"점수를 찾을 수 없습니다: {text!r}")
    value = float(m.group(0))
    if not 0 <= value <= 1:
        raise ValueError(f"점수 범위 오류: {value}")
    return round(value, 2)

def load_abilities():
    df = pd.read_csv(ABILITY_FILE, dtype=str).fillna("")
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    result = {}
    for code, group in df.groupby("konetOccpCd"):
        result[code] = {}
        for metric, limit in [("importance_score", TOP_N_IMPORTANCE), ("level_score", TOP_N_LEVEL)]:
            subset = group[group["metric_type"] == metric].sort_values("value_num", ascending=False).head(limit)
            result[code][metric] = [
                {"name": row["ability_name"], "value": float(row["value_num"])}
                for _, row in subset.iterrows()
            ]
    return result

def format_items(items):
    if not items:
        return "- 정보 없음"
    return "\n".join(f"- {x['name']}: {x['value']:.1f}" for x in items)

def build_prompt(row, ability):
    return f'''직업명: {row.get("konetOcnm", "")}

직무개요:
{row.get("dtyOtlnCn", "")}

중요도가 높은 업무수행능력:
{format_items(ability.get("importance_score", []))}

높은 수준이 요구되는 업무수행능력:
{format_items(ability.get("level_score", []))}

이 직업에서 생성형 AI를 활용할 경우 전체 작업시간이 평균적으로 얼마나 줄어드는지 평가하세요.
0.00~1.00 사이의 숫자 하나만 출력하세요.'''

def is_rate_limit_error(exc):
    text = str(exc).lower()
    return "429" in text or "rate_limit" in text or "tokens per min" in text or "requests per min" in text

def evaluate_one(client, prompt):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=256,
                reasoning={"effort": "minimal"},
            )
            raw = response.output_text.strip()
            return parse_score(raw), raw, response.id
        except Exception as exc:
            last_error = exc
            if not is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            wait_seconds = 25 + (attempt - 1) * 15
            print(f"  사용량 제한으로 {wait_seconds}초 대기 후 재시도 {attempt}/{MAX_RETRIES}")
            time.sleep(wait_seconds)
    raise last_error

def load_existing():
    if not OUTPUT_FILE.exists():
        return []
    return pd.read_csv(OUTPUT_FILE, dtype=str).fillna("").to_dict("records")

def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. 현재 PowerShell에서 다시 설정하세요.")
    if not DETAIL_FILE.exists():
        raise FileNotFoundError(DETAIL_FILE)
    if not ABILITY_FILE.exists():
        raise FileNotFoundError(ABILITY_FILE)

    # SDK 자체 자동 재시도를 끕니다. 그렇지 않으면 한 호출이 여러 RPM을 소모할 수 있습니다.
    client = OpenAI(max_retries=0)
    jobs = pd.read_csv(DETAIL_FILE, dtype=str).fillna("")
    abilities = load_abilities()

    print("기존 요청 제한이 초기화되도록 65초 기다린 뒤 시작합니다.")
    time.sleep(65)

    result_rows = load_existing()
    completed = {r["konetOccpCd"] for r in result_rows if r.get("status") == "완료"}
    error_rows = []
    total = len(jobs)

    for index, row in jobs.iterrows():
        code = row["konetOccpCd"]
        name = row["konetOcnm"]

        if code in completed:
            print(f"[{index + 1}/{total}] 건너뜀: {name}")
            continue

        print(f"[{index + 1}/{total}] 평가: {name}")
        prompt = build_prompt(row, abilities.get(code, {}))

        try:
            score, raw, response_id = evaluate_one(client, prompt)
            result_rows = [r for r in result_rows if r.get("konetOccpCd") != code]
            result_rows.append({
                "konetOccpCd": code,
                "konetOcnm": name,
                "model": MODEL,
                "AI_score": f"{score:.2f}",
                "raw_output": raw,
                "response_id": response_id,
                "status": "완료",
            })
            save_csv(result_rows, OUTPUT_FILE)
            print(f"  점수: {score:.2f}")
        except Exception as exc:
            error_rows.append({
                "konetOccpCd": code,
                "konetOcnm": name,
                "model": MODEL,
                "reason": str(exc),
            })
            save_csv(error_rows, ERROR_FILE)
            print(f"  오류: {exc}")

        if index < total - 1:
            print(f"  다음 요청까지 {REQUEST_INTERVAL_SECONDS}초 대기")
            time.sleep(REQUEST_INTERVAL_SECONDS)

    print("\n완료")
    print("결과:", OUTPUT_FILE)
    if ERROR_FILE.exists():
        print("오류:", ERROR_FILE)

if __name__ == "__main__":
    main()
