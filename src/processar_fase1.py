"""
Fase 1 - transforma o RouterBench bruto (routerbench_0shot.pkl) num CSV de
trabalho com uma linha por pergunta, comparando o modelo pequeno
(gpt-3.5-turbo-1106) e o grande (gpt-4-1106-preview).

Decisões tomadas com a usuária (ver conversa da Fase 1):
- Acerto = score == 1.0 (critério rigoroso; scores fracionários contam como erro)
- Subtópicos do MMLU e do MT-Bench são agrupados numa única categoria cada
- Categorias fora da lista oficial do paper (tarefas em chinês, consensus_summary,
  bias_detection, abstract2title, accounting_audit, test-match) são descartadas
- Par de modelos: gpt-3.5-turbo-1106 (pequeno) vs gpt-4-1106-preview (grande)
"""

import ast
from pathlib import Path

import pandas as pd

ORIGEM = Path(__file__).resolve().parent.parent / "dados" / "bruto" / "routerbench_0shot.pkl"
DESTINO = Path(__file__).resolve().parent.parent / "dados" / "processado" / "routerbench_fase1.csv"

MODELO_PEQUENO = "gpt-3.5-turbo-1106"
MODELO_GRANDE = "gpt-4-1106-preview"


def mapear_categoria(eval_name: str) -> str | None:
    """Mapeia o eval_name original para a categoria final, ou None se a categoria
    estiver fora do escopo do projeto (não faz parte da lista oficial do paper)."""
    if eval_name.startswith("mmlu-"):
        return "mmlu"
    if eval_name.startswith("mtbench"):
        return "mtbench"
    if eval_name == "grade-school-math":
        return "gsm8k"
    if eval_name in ("hellaswag", "arc-challenge", "winogrande", "mbpp"):
        return eval_name
    return None


def extrair_texto_pergunta(prompt_serializado: str) -> str:
    """O campo 'prompt' do dataset é uma string no formato de lista Python
    (ex: "['instrução', 'pergunta']"), não uma lista de verdade. Aqui desserializamos
    com ast.literal_eval e juntamos os elementos numa única string de texto."""
    partes = ast.literal_eval(prompt_serializado)
    return "\n\n".join(partes)


def processar():
    df = pd.read_pickle(ORIGEM)
    n_original = len(df)

    # 1. remover prompts duplicados (mantém a primeira ocorrência)
    df = df.drop_duplicates(subset=["prompt"], keep="first")
    n_sem_duplicatas = len(df)

    # 2. mapear categorias e descartar as que ficam fora do escopo
    df["categoria"] = df["eval_name"].apply(mapear_categoria)
    df = df[df["categoria"].notna()]
    n_apos_filtro_categoria = len(df)

    # 3. desserializar o texto da pergunta
    df["pergunta"] = df["prompt"].apply(extrair_texto_pergunta)

    # 4. acerto binário: só conta como acerto se o score for exatamente 1.0
    df["acerto_pequeno"] = (df[MODELO_PEQUENO] == 1.0).astype(int)
    df["acerto_grande"] = (df[MODELO_GRANDE] == 1.0).astype(int)

    # 5. custos (em dólares, calculados a partir de tabela de preço público)
    df["custo_pequeno"] = df[f"{MODELO_PEQUENO}|total_cost"]
    df["custo_grande"] = df[f"{MODELO_GRANDE}|total_cost"]

    df_final = df[["pergunta", "categoria", "acerto_pequeno", "acerto_grande", "custo_pequeno", "custo_grande"]]

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(DESTINO, index=False, encoding="utf-8")

    print(f"Linhas no dataset original:              {n_original}")
    print(f"Após remover duplicatas:                 {n_sem_duplicatas} (-{n_original - n_sem_duplicatas})")
    print(f"Após filtrar categorias fora de escopo:   {n_apos_filtro_categoria} (-{n_sem_duplicatas - n_apos_filtro_categoria})")
    print()
    print("Distribuição final por categoria:")
    print(df_final["categoria"].value_counts())
    print()
    print(f"Taxa de acerto do modelo pequeno ({MODELO_PEQUENO}): {df_final['acerto_pequeno'].mean():.1%}")
    print(f"Taxa de acerto do modelo grande  ({MODELO_GRANDE}):  {df_final['acerto_grande'].mean():.1%}")
    print()
    print(f"CSV salvo em: {DESTINO}")


if __name__ == "__main__":
    processar()
