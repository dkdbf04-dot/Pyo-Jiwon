
"""
오류가 난 직업만 GPT-5로 재평가한 뒤 기존 결과와 합쳐
최종 461개 직업 데이터셋을 만드는 코드입니다.

같은 폴더에 필요한 파일:
1. GPT5_직업별_AI시간절감점수.csv
2. GPT5_직업별_AI시간절감점수_오류.csv
3. 임금직업포털_직업상세.csv
4. 임금직업포털_업무수행능력_긴형식.csv

PowerShell에서 API 키 설정:
    $env:OPENAI_API_KEY="새_API_키"

실행:
    python GPT5_오류직업_재평가_최종병합.py

출력:
- GPT5_오류직업_재평가결과.csv
- GPT5_직업별_AI시간절감점수_최종.csv
- GPT5_오류직업_재평가실패.csv   (실패가 있을 때만 생성)
"""

import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE = Path(__file__).resolve().parent

RAW_RESULT_FILE = BASE / "GPT5_직업별_AI시간절감점수.csv"
ERROR_FILE = BASE / "GPT5_직업별_AI시간절감점수_오류.csv"
DETAIL_FILE = BASE / "임금직업포털_직업상세.csv"
ABILITY_FILE = BASE / "임금직업포털_업무수행능력_긴형식.csv"

RETRY_RESULT_FILE = BASE / "GPT5_오류직업_재평가결과.csv"
RETRY_FAILURE_FILE = BASE / "GPT5_오류직업_재평가실패.csv"
FINAL_FILE = BASE / "GPT5_직업별_AI시간절감점수_최종.csv"

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TOP_N_IMPORTANCE = 6
TOP_N_LEVEL = 6
REQUEST_INTERVAL_SECONDS = 3
MAX_RETRIES = 5

SYSTEM_PROMPT = """당신은 직업의 특성과 각각의 직업이 요구하는 숙련을 잘 아는 전문가입니다.
당신은 생성형 AI의 기능과 한계도 잘 알고 있습니다.
주어진 직업에서 현재의 생성형 AI를 보조 도구로 활용할 때 전체 작업시간이 평균적으로 얼마나 줄어드는지 평가하세요.

평가기준:
- 0.00: AI로 작업시간을 거의 줄일 수 없음
- 1.00: 해당 직업의 전체 업무를 AI가 사실상 전적으로 수행할 수 있음
- 문서작성, 요약, 정보검색, 분석, 번역, 코딩은 시간절감 가능성이 높음
- 현장 이동, 육체노동, 장비 조작, 대면 돌봄, 최종 법적·윤리적 책임은 낮음
- 직업 전체의 평균 업무시간을 기준으로 판단
- 설명 없이 0.00~1.00 사이의 숫자 하나만 출력
"""


def parse_score(text: str) -> float:
    match = re.search(
        r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)",
        text.strip(),
    )
    if not match:
        raise ValueError(f"점수를 찾을 수 없습니다: {text!r}")

    score = float(match.group(0))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"점수 범위 오류: {score}")

    return round(score, 2)


def load_abilities() -> dict:
    ability_df = pd.read_csv(ABILITY_FILE, dtype=str).fillna("")
    ability_df["value_num"] = pd.to_numeric(
        ability_df["value"], errors="coerce"
    ).fillna(0.0)

    result = {}

    for code, group in ability_df.groupby("konetOccpCd"):
        result[code] = {}

        for metric, limit in [
            ("importance_score", TOP_N_IMPORTANCE),
            ("level_score", TOP_N_LEVEL),
        ]:
            subset = (
                group[group["metric_type"] == metric]
                .sort_values("value_num", ascending=False)
                .head(limit)
            )

            result[code][metric] = [
                {
                    "name": row["ability_name"],
                    "value": float(row["value_num"]),
                }
                for _, row in subset.iterrows()
            ]

    return result


def format_items(items: list[dict]) -> str:
    if not items:
        return "- 정보 없음"

    return "\n".join(
        f"- {item['name']}: {item['value']:.1f}"
        for item in items
    )


def build_prompt(row: pd.Series, ability: dict) -> str:
    return f"""직업명: {row.get('konetOcnm', '')}

직무개요:
{row.get('dtyOtlnCn', '')}

중요도가 높은 업무수행능력:
{format_items(ability.get('importance_score', []))}

높은 수준이 요구되는 업무수행능력:
{format_items(ability.get('level_score', []))}

이 직업에서 생성형 AI를 활용할 경우 전체 작업시간이 평균적으로 얼마나 줄어드는지 평가하세요.
0.00~1.00 사이의 숫자 하나만 출력하세요."""


def evaluate_one(
    client: OpenAI,
    prompt: str,
) -> tuple[float, str, str]:
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

            raw_output = response.output_text.strip()
            score = parse_score(raw_output)
            return score, raw_output, response.id

        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            wait_seconds = 10 * attempt
            print(
                f"  오류 발생: {exc}\n"
                f"  {wait_seconds}초 후 재시도 "
                f"{attempt + 1}/{MAX_RETRIES}"
            )
            time.sleep(wait_seconds)

    raise last_error


def clean_existing_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    같은 직업이 여러 번 저장된 경우:
    1. repeat_no를 숫자로 변환
    2. 직업별로 repeat_no가 가장 큰 행을 선택
    3. repeat_no가 같으면 파일에서 마지막 행을 선택
    """
    df = raw_df.copy()
    df["_row_order"] = range(len(df))
    df["_repeat_num"] = pd.to_numeric(
        df.get("repeat_no", "0"),
        errors="coerce",
    ).fillna(0)

    df = (
        df.sort_values(
            ["konetOccpCd", "_repeat_num", "_row_order"]
        )
        .drop_duplicates(
            subset=["konetOccpCd"],
            keep="last",
        )
        .drop(columns=["_row_order", "_repeat_num"])
        .reset_index(drop=True)
    )

    return df


def save_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "PowerShell에서 API 키를 설정하세요."
        )

    required_files = [
        RAW_RESULT_FILE,
        ERROR_FILE,
        DETAIL_FILE,
        ABILITY_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"필요한 파일이 없습니다: {file_path.name}"
            )

    client = OpenAI(max_retries=0)

    raw_results = pd.read_csv(
        RAW_RESULT_FILE,
        dtype=str,
    ).fillna("")

    existing_clean = clean_existing_results(raw_results)

    error_df = pd.read_csv(
        ERROR_FILE,
        dtype=str,
    ).fillna("")

    # 오류파일에 중복 직업이 있더라도 직업코드별 한 번만 재평가
    error_df = error_df.drop_duplicates(
        subset=["konetOccpCd"],
        keep="last",
    )

    detail_df = pd.read_csv(
        DETAIL_FILE,
        dtype=str,
    ).fillna("")

    abilities = load_abilities()

    detail_by_code = (
        detail_df
        .drop_duplicates("konetOccpCd", keep="last")
        .set_index("konetOccpCd")
    )

    retry_rows = []
    failure_rows = []

    total = len(error_df)

    for number, (_, error_row) in enumerate(
        error_df.iterrows(),
        start=1,
    ):
        code = error_row["konetOccpCd"]
        name = error_row["konetOcnm"]

        print(f"[{number}/{total}] 재평가: {name}")

        if code not in detail_by_code.index:
            reason = "직업상세 파일에서 직업코드를 찾지 못함"
            print(f"  실패: {reason}")
            failure_rows.append(
                {
                    "konetOccpCd": code,
                    "konetOcnm": name,
                    "model": MODEL,
                    "reason": reason,
                }
            )
            continue

        detail_row = detail_by_code.loc[code]
        prompt = build_prompt(
            detail_row,
            abilities.get(code, {}),
        )

        try:
            score, raw_output, response_id = evaluate_one(
                client,
                prompt,
            )

            result = {
                "konetOccpCd": code,
                "konetOcnm": detail_row.get(
                    "konetOcnm",
                    name,
                ),
                "repeat_no": "retry",
                "model": MODEL,
                "AI_score": f"{score:.2f}",
                "raw_output": raw_output,
                "response_id": response_id,
                "status": "완료",
            }

            retry_rows.append(result)
            save_csv(retry_rows, RETRY_RESULT_FILE)
            print(f"  점수: {score:.2f}")

        except Exception as exc:
            print(f"  최종 실패: {exc}")
            failure_rows.append(
                {
                    "konetOccpCd": code,
                    "konetOcnm": name,
                    "model": MODEL,
                    "reason": str(exc),
                }
            )
            save_csv(
                failure_rows,
                RETRY_FAILURE_FILE,
            )

        if number < total:
            time.sleep(REQUEST_INTERVAL_SECONDS)

    retry_df = pd.DataFrame(retry_rows)

    if retry_df.empty:
        print("\n재평가에 성공한 직업이 없습니다.")
        return

    # 재평가된 직업은 기존 결과에서 제거한 뒤 새 결과를 붙임
    existing_without_retried = existing_clean[
        ~existing_clean["konetOccpCd"].isin(
            retry_df["konetOccpCd"]
        )
    ].copy()

    final_df = pd.concat(
        [existing_without_retried, retry_df],
        ignore_index=True,
    )

    final_df = (
        final_df
        .drop_duplicates(
            subset=["konetOccpCd"],
            keep="last",
        )
        .sort_values("konetOccpCd")
        .reset_index(drop=True)
    )

    final_df.to_csv(
        FINAL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n완료")
    print(f"기존 고유 직업 수: {len(existing_clean)}")
    print(f"오류 직업 수: {len(error_df)}")
    print(f"재평가 성공 수: {len(retry_df)}")
    print(f"재평가 실패 수: {len(failure_rows)}")
    print(f"최종 고유 직업 수: {final_df['konetOccpCd'].nunique()}")
    print(f"최종 파일: {FINAL_FILE}")

    if final_df["konetOccpCd"].nunique() == 461:
        print("정상: 461개 직업이 모두 포함되었습니다.")
    else:
        print(
            "주의: 최종 직업 수가 461개가 아닙니다. "
            "재평가실패 파일을 확인하세요."
        )


if __name__ == "__main__":
    main()
