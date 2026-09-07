"""
Baixa o dataset RouterBench (variante zero-shot) do Hugging Face.

Fonte: https://huggingface.co/datasets/withmartian/routerbench
Licença do código associado (paper): MIT (github.com/withmartian/routerbench)
Citação: Hu et al., "RouterBench: A Benchmark for Multi-LLM Routing System", ICML 2024
"""

import requests
from pathlib import Path

URL = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl"
DESTINO = Path(__file__).resolve().parent.parent / "dados" / "bruto" / "routerbench_0shot.pkl"


def baixar():
    if DESTINO.exists():
        tamanho_mb = DESTINO.stat().st_size / (1024 * 1024)
        print(f"Arquivo já existe em {DESTINO} ({tamanho_mb:.1f} MB). Nada a fazer.")
        return

    print(f"Baixando de {URL} ...")
    resposta = requests.get(URL, stream=True, timeout=60)
    resposta.raise_for_status()

    total_bytes = int(resposta.headers.get("content-length", 0))
    baixado = 0

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    with open(DESTINO, "wb") as arquivo:
        for pedaco in resposta.iter_content(chunk_size=1024 * 1024):  # 1 MB por vez
            arquivo.write(pedaco)
            baixado += len(pedaco)
            if total_bytes:
                percentual = 100 * baixado / total_bytes
                print(f"\r{baixado / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({percentual:.0f}%)", end="")

    print(f"\nConcluído: {DESTINO}")


if __name__ == "__main__":
    baixar()
