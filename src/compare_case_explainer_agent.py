#!/usr/bin/env python3
"""
Agente de explicação (DeepSeek) para cada dia/caso de comparação modelo vs realidade.

Lê ``metrics_by_day.json`` gerado por ``compare_st_hypernet_days`` ou
``compare_goes_unsupervised_days`` e, para cada entrada (ou um subconjunto),
chama a API **DeepSeek** (compatível com OpenAI) para produzir texto em português
sobre o que ocorreu e como o modelo se comportou face aos focos reais na grade.

Chave de API: variável de ambiente ``DEEPSEEK_API_KEY`` (recomendado com ``.env``)
ou ``--api-key`` (evitar em scripts partilhados).

Exemplo::

    export DEEPSEEK_API_KEY=sk-...
    python -m src.compare_case_explainer_agent \\
        --metrics-dir data/st_hypernet_2024_all_fire_days \\
        --out data/st_hypernet_2024_all_fire_days/case_explanations.json \\
        --max-cases 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(_repo_root / ".env")
    load_dotenv()
except ImportError:
    pass

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT_PT = """\
És um analista de ciência de dados especializado em detecção de queimadas no Ceará, Brasil.
Recebes métricas de UM dia: grade espacial (células) com focos reais vs células previstas por um modelo
(ST-HyperNet ou GOES não supervisionado; cubo derivado de focos — métricas exploratórias).

O utilizador prioriza **precisão** (acerto das células previstas: TP/(TP+FP)) e o custo de **falsos alarmes (FP)**.
Trata o **recall** apenas como contexto de cobertura (TP/(TP+FN)), sem o colocar como objectivo principal.

Escreve em **português europeu ou brasileiro claro**, em 3 a 6 parágrafos curtos:
1) Precisão, FP e TP em linguagem acessível; depois IoU; recall só se for relevante para o caso.
2) Se o modelo foi sobretudo **alarmista** (muitos FP → precisão baixa) ou **conservador nas previsões de célula** (poucos FP mas possivelmente muitos FN).
3) Como interpretar o limiar adaptativo (se presente) vs focos reais (n_focos).
4) Limitações: grade grosseira, proxy GOES, não substitui validação operacional INPE.

Não inventes dados que não estejam no JSON. Não cries coordenadas ou datas novas.
"""


def _read_metrics_bundle(metrics_path: Path) -> Dict[str, Any]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = data.get("metrics_by_day")
    if not isinstance(rows, list):
        raise ValueError("JSON sem lista 'metrics_by_day'.")
    return data


def slim_case_for_llm(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reduz o registo do dia a campos úteis para o LLM (menos tokens, sem ruído)."""
    keys = (
        "date",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "n_focos",
        "n_cube_frames_that_day",
        "pred_threshold_base",
        "pred_threshold",
        "pred_norm",
        "pred_display_mode",
        "pred_display_threshold",
    )
    out: Dict[str, Any] = {k: row[k] for k in keys if k in row}
    meta = row.get("adaptive_threshold_meta")
    if isinstance(meta, dict):
        out["adaptive_threshold_meta"] = {
            k: meta[k]
            for k in ("mode", "chosen", "objective_score", "adaptive_model", "relaxed_min_recall")
            if k in meta
        }
    return out


def _context_header(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "technique": bundle.get("technique"),
        "caveat": bundle.get("caveat"),
        "year_filter": bundle.get("year_filter"),
        "grid_shape": bundle.get("grid_shape"),
        "pred_threshold_global": bundle.get("pred_threshold"),
        "adaptive_threshold_config": bundle.get("adaptive_threshold"),
        "cube_time_span": bundle.get("cube_time_span"),
    }


def deepseek_chat_completion(
    api_key: str,
    user_content: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.35,
    timeout_sec: int = 120,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_PT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    r = requests.post(
        DEEPSEEK_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_sec,
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Resposta DeepSeek sem choices: {data}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not content or not isinstance(content, str):
        raise RuntimeError(f"Resposta DeepSeek sem content: {data}")
    return content.strip()


def run(
    metrics_dir: Path,
    out_path: Path,
    api_key: str,
    max_cases: int = 0,
    dates: Optional[Sequence[str]] = None,
    model: str = DEFAULT_MODEL,
    sleep_sec: float = 0.45,
    dry_run: bool = False,
) -> Dict[str, Any]:
    metrics_file = metrics_dir / "metrics_by_day.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Não encontrado: {metrics_file}")

    bundle = _read_metrics_bundle(metrics_file)
    rows: List[Dict[str, Any]] = bundle["metrics_by_day"]
    date_set = None
    if dates:
        date_set = {d.strip() for d in dates if d.strip()}

    selected: List[Dict[str, Any]] = []
    for row in rows:
        d = str(row.get("date", ""))
        if date_set is not None and d not in date_set:
            continue
        selected.append(row)
        if max_cases and len(selected) >= int(max_cases):
            break

    if not selected:
        raise ValueError("Nenhum dia seleccionado (verifique --dates ou --max-cases).")

    header = _context_header(bundle)
    cases_out: List[Dict[str, Any]] = []

    for i, row in enumerate(selected):
        slim = slim_case_for_llm(row)
        user_payload = {
            "contexto_geral_do_relatorio": header,
            "caso_do_dia": slim,
        }
        user_txt = json.dumps(user_payload, ensure_ascii=False, indent=2)

        if dry_run:
            explanation = "[dry-run] Pedido não enviado à API."
        else:
            explanation = deepseek_chat_completion(api_key, user_txt, model=model)
            if i + 1 < len(selected) and sleep_sec > 0:
                time.sleep(float(sleep_sec))

        cases_out.append(
            {
                "date": row.get("date"),
                "explanation_pt": explanation,
                "input_digest": slim,
                "model": model if not dry_run else "dry-run",
            }
        )

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_metrics_file": str(metrics_file),
        "deepseek_model": model,
        "n_cases": len(cases_out),
        "cases": cases_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _parse_dates_arg(s: str | None) -> Optional[List[str]]:
    if not s or not str(s).strip():
        return None
    parts = re.split(r"[,\s;]+", str(s).strip())
    return [p for p in parts if p]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Explica cada caso (dia) modelo vs realidade via API DeepSeek."
    )
    p.add_argument(
        "--metrics-dir",
        type=str,
        required=True,
        help="Pasta com metrics_by_day.json (ex.: data/st_hypernet_2024_all_fire_days)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Ficheiro JSON de saída (default: <metrics-dir>/case_explanations.json)",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Chave DeepSeek (preferir variável DEEPSEEK_API_KEY)",
    )
    p.add_argument("--max-cases", type=int, default=0, help="Máximo de dias (0 = todos os seleccionados)")
    p.add_argument(
        "--dates",
        type=str,
        default="",
        help="Datas YYYY-MM-DD separadas por vírgula (opcional; filtra antes de --max-cases)",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo DeepSeek (ex.: deepseek-chat)")
    p.add_argument("--sleep", type=float, default=0.45, help="Pausa entre pedidos (segundos)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Monta payloads e lista dias sem chamar a API",
    )
    args = p.parse_args()

    mdir = Path(args.metrics_dir)
    out_p = Path(args.out) if str(args.out).strip() else mdir / "case_explanations.json"
    key = (args.api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not args.dry_run and not key:
        raise SystemExit(
            "Defina DEEPSEEK_API_KEY no ambiente ou passe --api-key (não commite chaves)."
        )

    dates = _parse_dates_arg(args.dates or None)
    max_c = int(args.max_cases) if int(args.max_cases) > 0 else 0

    res = run(
        metrics_dir=mdir,
        out_path=out_p,
        api_key=key or "dry",
        max_cases=max_c,
        dates=dates,
        model=str(args.model),
        sleep_sec=float(args.sleep),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({"n_cases": res["n_cases"], "out": str(out_p)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
