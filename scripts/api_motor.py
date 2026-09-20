"""Extrai e classifica candidatos de valores quantitativos em abstracts sobre REE.

Recebe o CSV filtrado pelo regex_motor.py (colunas Abstract_id, Abstract): o
modelo lê o abstract inteiro, encontra os valores candidatos e já os
classifica, via few-shot. O resultado é salvo incrementalmente em um .jsonl
(uma linha por abstract), no esquema sugerido pelo professor: cada linha traz
o índice `i`, então relendo o arquivo dá para saber exatamente onde parou e
rodar de novo só o que falta.

Exemplo de chamada:

python api_motor.py --input ..\outputs\abstracts_candidatos.csv --output ..\outputs\extracoes.jsonl --limite 5 --consolidar

Autor: Edélio Gabriel Magalhães de Jesus

Desenvolvido com auxílio de Inteligência Artificial.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH_PADRAO = BASE_DIR / "outputs" / "abstracts_candidatos.csv"
SAIDA_PADRAO = BASE_DIR / "outputs" / "extracoes.jsonl"

BASE_URL = "https://iluma.cnpem.br:4000/v1"
MODELO = "iluma"
TEMPERATURE = 0.5
MAX_TOKEN = 4096


SYSTEM_PROMPT = """
You are an expert data extractor analyzing scientific abstracts about rare-earth elements (REE). Your task is to find every candidate quantitative value related to REE metrics in the abstract below and classify each one.

Two structural facts about this domain matter a lot: (1) values are very often reported as a RANGE (min-max) rather than a single number, and (2) the entity being measured is very often a GROUP, OXIDE, or MINERAL/ALLOY rather than a single chemical element. Your extraction must capture both of these explicitly instead of flattening them.

For each candidate value you find, return a JSON object with the following fields:
- "value": (string) The value exactly as it appears in the text. If the value is a range, this is the MINIMUM/lower bound (e.g., "36 ppm" in "from 36 to 339 ppm"). If it is a single value, this is that value (e.g., "971 ppm"). Do not change or normalize it.
- "value_max": (string or null) ONLY set when "is_range" is true: the MAXIMUM/upper bound of the range, exactly as it appears (e.g., "339 ppm" in "from 36 to 339 ppm"). Set to null when "is_range" is false.
- "is_range": (boolean) true if the text expresses a min-max range (e.g., "from 36 to 339 ppm", "20% to 70%", "339-971 ppm"), false if it is a single/average/maximum-only/minimum-only reported value.
- "sentence": (string) The sentence (or short excerpt) where the value appears, quoted from the abstract.
- "metric_type": (string) Classify the value into ONE of the following categories:
    * "Bulk_Concentration": Single, average, total, or range values for AGGREGATED REE/TREO/REO/ΣREE (e.g., "971 ppm TREO", or the range "36 to 339 ppm" of total REE). Use this regardless of whether it's a single value or a range — "is_range" already captures that distinction.
    * "Individual_Entity_Concentration": Concentrations or grades of a SPECIFIC entity: a single REE element (Nd, Y, Sc, Dy...), a specific mineral (bastnaesite, monazite, xenotime...), or a specific alloy/compound. Use this for both single values and ranges of a specific named entity.
    * "Process_Metric": Values related to extraction, recovery efficiency, leaching, selectivity, enrichment ratio, or laboratory solutions.
    * "Invalid_Candidate": Values completely unrelated to REE metrics (e.g., general water salinity, temperature, non-REE elements like uranium unless explicitly about REE co-occurrence).
- "target_entity": (string) The specific element, oxide, group, or mineral being measured (e.g., "TREO", "Nd", "Total REE", "Heavy REE", "Bastnaesite", "TDS").
- "entity_type": (string) Classify "target_entity" into ONE of: "element" (single chemical element, e.g. Nd, Y, Dy), "element_group" (aggregated group, e.g. Light REE, Heavy REE, Total REE), "oxide" (e.g. TREO, REO), "mineral" (specific named mineral/alloy, e.g. bastnaesite, monazite, xenotime), "other" (anything else, e.g. TDS, gangue minerals — typically paired with Invalid_Candidate).
- "context_modifier": (string) Any text modifier defining the value (e.g., "average", "maximum", "greater than (>)", "approximate (~)", "none"). For ranges, this is usually "none" since "is_range" already conveys the range nature.

Rules:
- Only extract values that are actual measured/reported quantities (concentrations, grades, ratios, recoveries, percentages, etc.), not dates, sample counts, or coordinates.
- A min-max range is ONE extraction with "is_range": true and both "value" and "value_max" filled in — never split a single range into two separate extractions.
- If the abstract has no relevant candidate values, return an empty list.
- Return valid JSON only, in the shape {"extracoes": [...]}. Do not include any text outside the JSON object.
""".strip()


FEW_SHOT = [
    {
        "input": {
            "abstract": "The total REE content in this ore was ~6720 ppm (~3540 ppm light REE and ~3180 ppm heavy REE)."
        },
        "output": {
            "extracoes": [
                {
                    "value": "6720 ppm",
                    "value_max": None,
                    "is_range": False,
                    "sentence": "The total REE content in this ore was ~6720 ppm (~3540 ppm light REE and ~3180 ppm heavy REE).",
                    "metric_type": "Bulk_Concentration",
                    "target_entity": "Total REE",
                    "entity_type": "element_group",
                    "context_modifier": "approximate (~)",
                },
                {
                    "value": "3540 ppm",
                    "value_max": None,
                    "is_range": False,
                    "sentence": "The total REE content in this ore was ~6720 ppm (~3540 ppm light REE and ~3180 ppm heavy REE).",
                    "metric_type": "Bulk_Concentration",
                    "target_entity": "Light REE",
                    "entity_type": "element_group",
                    "context_modifier": "approximate (~)",
                },
            ]
        },
    },
    {
        "input": {
            "abstract": "The rocks have total REE contents within the range of 36 to 339 ppm."
        },
        "output": {
            "extracoes": [
                {
                    "value": "36 ppm",
                    "value_max": "339 ppm",
                    "is_range": True,
                    "sentence": "The rocks have total REE contents within the range of 36 to 339 ppm.",
                    "metric_type": "Bulk_Concentration",
                    "target_entity": "Total REE",
                    "entity_type": "element_group",
                    "context_modifier": "none",
                }
            ]
        },
    },
    {
        "input": {
            "abstract": "High-grade samples have economically relevant REE accumulations (Nd > 30,000 ppm)."
        },
        "output": {
            "extracoes": [
                {
                    "value": "30,000 ppm",
                    "value_max": None,
                    "is_range": False,
                    "sentence": "High-grade samples have economically relevant REE accumulations (Nd > 30,000 ppm).",
                    "metric_type": "Individual_Entity_Concentration",
                    "target_entity": "Nd",
                    "entity_type": "element",
                    "context_modifier": "greater than (>)",
                }
            ]
        },
    },
    {
        "input": {
            "abstract": "Bastnaesite in the deposit contains between 60% and 75% REO, making it the dominant ore mineral."
        },
        "output": {
            "extracoes": [
                {
                    "value": "60%",
                    "value_max": "75%",
                    "is_range": True,
                    "sentence": "Bastnaesite in the deposit contains between 60% and 75% REO, making it the dominant ore mineral.",
                    "metric_type": "Individual_Entity_Concentration",
                    "target_entity": "Bastnaesite",
                    "entity_type": "mineral",
                    "context_modifier": "none",
                }
            ]
        },
    },
    {
        "input": {
            "abstract": "REE biosorption has high recovery efficiency with TDS as high as 165,000 ppm."
        },
        "output": {
            "extracoes": [
                {
                    "value": "165,000 ppm",
                    "value_max": None,
                    "is_range": False,
                    "sentence": "REE biosorption has high recovery efficiency with TDS as high as 165,000 ppm.",
                    "metric_type": "Invalid_Candidate",
                    "target_entity": "TDS (Total Dissolved Solids)",
                    "entity_type": "other",
                    "context_modifier": "maximum limit",
                }
            ]
        },
    },
    {
        "input": {
            "abstract": "Mineral resources estimates have on average 145 Mt @ 971 ppm TREO."
        },
        "output": {
            "extracoes": [
                {
                    "value": "971 ppm",
                    "value_max": None,
                    "is_range": False,
                    "sentence": "Mineral resources estimates have on average 145 Mt @ 971 ppm TREO.",
                    "metric_type": "Bulk_Concentration",
                    "target_entity": "TREO",
                    "entity_type": "oxide",
                    "context_modifier": "average",
                }
            ]
        },
    },
    {
        "input": {
            "abstract": "The metallurgical test achieved a recovery rate of 65% to 85% for heavy rare earths."
        },
        "output": {
            "extracoes": [
                {
                    "value": "65%",
                    "value_max": "85%",
                    "is_range": True,
                    "sentence": "The metallurgical test achieved a recovery rate of 65% to 85% for heavy rare earths.",
                    "metric_type": "Process_Metric",
                    "target_entity": "Heavy REE",
                    "entity_type": "element_group",
                    "context_modifier": "none",
                }
            ]
        },
    },
]


def get_client() -> OpenAI:
    load_dotenv(BASE_DIR / "chave.env")
    token = os.getenv("ILUMA_API_KEY")
    if not token:
        raise RuntimeError("ILUMA_API_KEY não encontrada em chave.env.")

    return OpenAI(base_url=BASE_URL, api_key=token, timeout=120.0, max_retries=1)


def load_abstracts(caminho_csv: Path) -> list[dict]:
    """Lê o CSV (Abstract_id, Abstract) e devolve uma lista de itens a processar."""
    df = pd.read_csv(caminho_csv).fillna("")
    return [
        {"abstract_id": row["Abstract_id"], "texto": row["Abstract"]}
        for _, row in df.iterrows()
    ]


def extrair(client: OpenAI, texto: str) -> list[dict]:
    """Chama a API para um único abstract e devolve a lista de candidatos classificados."""
    prompt = {
        "task": "Find and classify every REE-related quantitative value candidate in this abstract.",
        "few_shot_examples": FEW_SHOT,
        "abstract": texto,
    }
    response = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKEN,
        response_format={"type": "json_object"},
    )

    if not response.choices or not response.choices[0].message.content:
        reason = response.choices[0].finish_reason if response.choices else "no choices"
        raise RuntimeError(f"A API não retornou conteúdo (finish_reason={reason}).")

    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise ValueError("A API retornou JSON inválido.") from exc

    extracoes = data.get("extracoes")
    if not isinstance(extracoes, list):
        raise ValueError("A API não devolveu uma lista de extrações.")

    return extracoes


def compactar_jsonl(saida: Path) -> None:
    """Reescreve `saida` mantendo uma única linha por índice `i`.

    Quando um item foi tentado mais de uma vez (ex.: erro e depois sucesso no
    retry), fica só a versão de sucesso mais recente; se só existir erro, fica
    o erro mais recente. Não faz nada se o arquivo não existir.
    """
    if not saida.exists():
        return

    registros: dict[int, dict] = {}
    with saida.open(encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            registro = json.loads(l)
            i = registro["i"]
            existente = registros.get(i)
            # sucesso sempre vence erro; entre dois sucessos (ou dois erros), fica o mais recente
            if existente is None or "erro" in existente or "erro" not in registro:
                registros[i] = registro

    with saida.open("w", encoding="utf-8") as f:
        for i in sorted(registros):
            f.write(json.dumps(registros[i], ensure_ascii=False) + "\n")


def rodar_lote(
    client: OpenAI,
    itens: list[dict],
    saida: Path,
    comeco: int = 0,
    limite: int | None = None,
) -> None:
    """Processa `itens` a partir de `comeco`, gravando uma linha por item em `saida`.

    Cada linha do .jsonl tem o índice `i`: relendo o arquivo você sabe exatamente
    onde parou, e roda de novo só o que falta. Duplicatas de execuções
    anteriores (ex.: erro seguido de sucesso no retry) são compactadas
    automaticamente antes de decidir o que falta processar.
    """
    saida.parent.mkdir(parents=True, exist_ok=True)
    compactar_jsonl(saida)

    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                registro = json.loads(l)
                if "erro" not in registro:
                    feitos.add(registro["i"])
        print(f"{len(feitos)} itens já processados com sucesso — serão pulados")

    alvos = list(enumerate(itens))[comeco:]
    if limite:
        alvos = alvos[:limite]

    with saida.open("a", encoding="utf-8") as f:
        for i, item in alvos:
            if i in feitos:
                continue
            try:
                linha = {
                    "i": i,
                    "abstract_id": item["abstract_id"],
                    "extracoes": extrair(client, item["texto"]),
                }
            except Exception as e:  # rede, timeout, cota
                linha = {"i": i, "abstract_id": item["abstract_id"], "erro": f"{type(e).__name__}: {e}"}
                print(f"  [{i}] falhou: {type(e).__name__}")
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()  # o segredo: gravar já
            time.sleep(0.2)

    compactar_jsonl(saida)
    print("fim")


def consolidar_csv(entrada_jsonl: Path, saida_csv: Path) -> pd.DataFrame:
    """Lê o .jsonl de extrações e achata tudo em um único CSV (uma linha por valor extraído)."""
    compactar_jsonl(entrada_jsonl)

    linhas = []
    with entrada_jsonl.open(encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            registro = json.loads(l)
            for extracao in registro.get("extracoes", []):
                linhas.append({"abstract_id": registro["abstract_id"], **extracao})

    df = pd.DataFrame(linhas)
    df.to_csv(saida_csv, index=False, encoding="utf-8")
    print(f"CSV consolidado salvo em: {saida_csv} ({len(df)} valores)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai e classifica valores quantitativos de REE em abstracts, via LLM."
    )
    parser.add_argument(
        "--input",
        default=str(CSV_PATH_PADRAO),
        help=f"Caminho do CSV de entrada (colunas Abstract_id, Abstract). Padrão: {CSV_PATH_PADRAO}",
    )
    parser.add_argument(
        "--output",
        default=str(SAIDA_PADRAO),
        help=f"Caminho do .jsonl de saída (retomável). Padrão: {SAIDA_PADRAO}",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=5,
        help="Quantos abstracts processar nesta execução (padrão: 5 — comece pequeno).",
    )
    parser.add_argument(
        "--comeco",
        type=int,
        default=0,
        help="Índice a partir do qual começar (padrão: 0).",
    )
    parser.add_argument(
        "--consolidar",
        action="store_true",
        help="Depois de processar o lote, gera também o CSV plano (uma linha por valor extraído).",
    )
    parser.add_argument(
        "--apenas-consolidar",
        action="store_true",
        help="Não chama a API: só lê o .jsonl existente (--output) e gera o CSV plano.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Caminho do CSV plano (usado com --consolidar ou --apenas-consolidar). "
        "Padrão: mesmo diretório do --output, arquivo extracoes_flat.csv.",
    )
    args = parser.parse_args()

    caminho_csv = Path(args.input)
    caminho_saida = Path(args.output)
    caminho_csv_plano = (
        Path(args.output_csv) if args.output_csv else caminho_saida.parent / "extracoes_flat.csv"
    )

    if args.apenas_consolidar:
        consolidar_csv(caminho_saida, caminho_csv_plano)
        return

    itens = load_abstracts(caminho_csv)
    if not itens:
        raise ValueError(f"Nenhum abstract encontrado em {caminho_csv}.")

    client = get_client()
    rodar_lote(client, itens, caminho_saida, comeco=args.comeco, limite=args.limite)

    if args.consolidar:
        consolidar_csv(caminho_saida, caminho_csv_plano)


if __name__ == "__main__":
    main()