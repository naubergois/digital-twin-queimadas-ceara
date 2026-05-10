#!/usr/bin/env python3
"""
Captura PNGs do dashboard Streamlit para docs/screenshots/ (README).

Requisitos (na raiz do repo):
  python -m venv .venv-capture && . .venv-capture/bin/activate
  pip install -r requirements.txt playwright
  python -m playwright install chromium

Executar:
  .venv-capture/bin/python scripts/capture_dashboard_screenshots.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "screenshots"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Instale: pip install playwright && python -m playwright install chromium", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO / "dashboard" / "app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    err_tail = ""
    try:
        # Espera HTTP do Streamlit (compilação da app pode demorar na 1ª vez)
        deadline = time.time() + 150
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=3)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                time.sleep(2)
                if proc.poll() is not None:
                    err_tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"Streamlit terminou antes de subir. stderr:\n{err_tail}")
        else:
            err_tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Timeout à espera de {url}. stderr:\n{err_tail}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
            page.locator('[data-testid="stAppViewContainer"]').wait_for(timeout=120000)
            page.locator('[role="tab"]').first.wait_for(timeout=120000)
            time.sleep(6)
            page.screenshot(path=str(OUT / "dashboard-aba-st-hypernet.png"), full_page=True)
            tabs = page.locator('[role="tab"]')
            if tabs.count() >= 2:
                tabs.nth(1).click()
            else:
                page.get_by_text("Focos no mapa", exact=True).click()
            time.sleep(5)
            page.screenshot(path=str(OUT / "dashboard-aba-focos.png"), full_page=True)
            browser.close()
    except Exception as ex:
        print(f"Falha na captura: {ex}", file=sys.stderr)
        return 2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"OK: {OUT / 'dashboard-aba-st-hypernet.png'}")
    print(f"OK: {OUT / 'dashboard-aba-focos.png'}")
    # PNGs de mapa/grade (achados) para o README — ignora falha se data/ não existir
    copy_script = REPO / "scripts" / "copy_sample_st_map_pngs_to_docs.py"
    if copy_script.is_file():
        r = subprocess.run([sys.executable, str(copy_script)], cwd=str(REPO))
        if r.returncode != 0:
            print("(aviso) copy_sample_st_map_pngs_to_docs.py não copiou PNGs — ver data/st_hypernet_*/")
    fig_script = REPO / "scripts" / "generate_readme_experiment_figures.py"
    if fig_script.is_file():
        subprocess.run([sys.executable, str(fig_script)], cwd=str(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
