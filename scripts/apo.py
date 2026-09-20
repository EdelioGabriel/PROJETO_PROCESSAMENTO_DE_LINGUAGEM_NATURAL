"""Otimizacao automatica de prompt (APO / ProTeGi) para extracao de REE.

Implementa o ciclo de "gradiente textual" descrito em Pryzant et al. 2023
(Automatic Prompt Optimization with "Gradient Descent" and Beam Search)
aplicado a extracao estruturada de valores quantitativos de REE, otimizando
um prompt simples/ingenuo (p0) e comparando o resultado contra o prompt
manual ja refinado do api_motor.py.

O gold-standard e dividido automaticamente 70/30 (treino/teste) de forma
deterministica (--seed, padrao 42): `otimizar` roda o ProTeGi so sobre o
TREINO; `avaliar-prompt` por padrao (--conjunto teste) avalia so sobre o
TESTE (holdout nunca visto na otimizacao), pra nao inflar o F1 reportado.
Use o MESMO --seed/--frac-treino nos dois comandos para reproduzir o
mesmo split.

Uso tipico:

  # 1. Rodar o ciclo de otimizacao ProTeGi a partir do p0 ingenuo (so no treino):
  python apo.py otimizar --gold gold_standard.jsonl --abstracts abstracts_candidatos.csv --passos 4 --output apo_run/

  # 2. Avaliar o prompt final produzido pelo APO no holdout de teste:
  python apo.py avaliar-prompt --prompt apo_run/prompt_final.txt --gold gold_standard.jsonl --abstracts abstracts_candidatos.csv --output apo_predicoes.jsonl

  # 3. Comparar com o manual (manual_predicoes.jsonl gerado como acima):
  python apo.py comparar --manual manual_predicoes.jsonl --apo apo_predicoes.jsonl --gold gold_standard.jsonl

  
python apo.py  otimizar --gold ../gold_standard/gold_standard.jsonl --abstracts ../outputs/abstracts_candidatos.csv --output ../outputs/output_apo  --passos 5 --beam 3 --seed 367 --frac-treino 0.7

python apo.py avaliar-prompt --prompt ../outputs/apo_run/prompt_final.txt --gold gold_standard.jsonl --abstracts ../outputs/abstracts_candidatos.csv --output ../outputs/apo_predicoes.jsonl
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_URL = "https://iluma.cnpem.br:4000/v1"
MODELO = "iluma"
TEMPERATURE_EXTRACAO = 0.5
TEMPERATURE_OTIMIZACAO = 1.0

# ============================================================
# Prompt manual (copiado do api_motor.py, mesmo schema) e o p0
# ingenuo usado como ponto de partida do APO.
# ============================================================

# p0 ingenuo: estrutura de OUTPUT fixa (nomes de campo, categorias
# permitidas), mas SEM nenhuma definicao do que cada categoria significa
# e sem few-shot. Isso e o que o ciclo ProTeGi deve preencher sozinho.
PROMPT_APO_INICIAL = """
Extract quantitative values related to rare-earth elements (REE) from the abstract below.

For each value found, return a JSON object with these fields, IN THIS ORDER: raciocinio, value, value_max, is_range, sentence, metric_type, target_entity, entity_type, context_modifier.

raciocinio: your reasoning for this specific candidate, written BEFORE the other fields (1-3 sentences). Explain what in the text made you consider this a valid REE-related quantitative value (or, if metric_type=Invalid_Candidate, why you are including-but-discarding it and what disqualified it), and why you chose this metric_type/entity_type over the other allowed options.

metric_type must be one of: Bulk_Concentration, Individual_Entity_Concentration, Process_Metric, Invalid_Candidate
entity_type must be one of: element, element_group, oxide, mineral, other

Return valid JSON only, in the shape {"extracoes": [...]}.
""".strip()


# ============================================================
# Cliente API (mesma infra do api_motor.py)
# ============================================================

def get_client() -> OpenAI:
    load_dotenv(BASE_DIR / "chave" / "chave.env")
    token = os.getenv("ILUMA_API_KEY")
    if not token:
        raise RuntimeError("ILUMA_API_KEY não encontrada em chave.env.")
    return OpenAI(base_url=BASE_URL, api_key=token, timeout=120.0, max_retries=1)


def chamar_llm(client: OpenAI, system_prompt: str, user_content: str,
                temperature: float = TEMPERATURE_EXTRACAO, json_mode: bool = True,
                habilitar_thinking: bool = False) -> str:
    """Chamada generica ao LLM. Retorna o texto da resposta (string).

    O modelo servido (Qwen3-class, MoE ~A10B) vem com "thinking" LIGADO por
    padrao no template de chat: ele gera um bloco de raciocinio interno
    ANTES do JSON de resposta, e esse raciocinio consome max_tokens. Com
    prompts mais elaborados (como os que o ProTeGi vai gerar a cada passo),
    isso e suficiente para estourar o teto e devolver finish_reason=length
    com content vazio -- que era exatamente o erro reportado. Por isso,
    desligamos o thinking por padrao aqui (habilitar_thinking=False) via
    o parametro extra_body/chat_template_kwargs suportado por vLLM para
    a familia Qwen3.
    """
    kwargs = dict(
        model=MODELO,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": habilitar_thinking}},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    if not response.choices or not response.choices[0].message.content:
        reason = response.choices[0].finish_reason if response.choices else "no choices"
        # Alguns backends (modelos com "thinking"/reasoning) consomem o
        # budget de max_tokens gerando um raciocinio interno antes do
        # conteudo final, e devolvem content vazio com finish_reason=length.
        # Se o backend expuser esse campo (reasoning_content, comum em
        # implementacoes estilo DeepSeek/vLLM/LiteLLM), isso confirma a causa.
        reasoning = getattr(response.choices[0].message, "reasoning_content", None) if response.choices else None
        usage = getattr(response, "usage", None)
        detalhes = f"finish_reason={reason}"
        if usage is not None:
            detalhes += f", usage={usage}"
        if reasoning:
            detalhes += (
                f", reasoning_content_tamanho={len(reasoning)} chars "
                "(o modelo pode estar gastando o max_tokens todo 'pensando' "
                "e nao sobra espaco para o JSON final -- considere reduzir o "
                "raciocinio ou aumentar MAX_TOKEN)"
            )
        raise RuntimeError(f"A API não retornou conteúdo ({detalhes}).")
    return response.choices[0].message.content


# ============================================================
# Extracao: roda um prompt (manual ou candidato do APO) sobre
# um conjunto de abstracts.
# ============================================================

def carregar_abstracts(caminho_csv: Path) -> dict:
    """Le o CSV (Abstract_id, Abstract) e devolve {abstract_id: texto}."""
    df = pd.read_csv(caminho_csv).fillna("")
    return {int(row["Abstract_id"]): row["Abstract"] for _, row in df.iterrows()}


def extrair_com_prompt(client: OpenAI, system_prompt: str, abstract_texto: str) -> list:
    """Roda o prompt (system_prompt) sobre um abstract e devolve a lista de
    extracoes (ja parseada), tolerando erro de parsing (devolve lista vazia
    e nao derruba o restante do lote)."""
    try:
        raw = chamar_llm(client, system_prompt, abstract_texto)
        data = json.loads(raw)
        extracoes = data.get("extracoes", [])
        if not isinstance(extracoes, list):
            return []
        return extracoes
    except Exception as e:
        print(f"    [erro na extracao: {type(e).__name__}: {e}]")
        return []


def rodar_prompt_sobre_abstracts(client: OpenAI, system_prompt: str,
                                   abstracts: dict, saida: Path) -> None:
    """Roda system_prompt sobre cada abstract em `abstracts` (dict id->texto)
    e grava incrementalmente em `saida` (.jsonl, uma linha por abstract_id),
    no mesmo espirito retomavel do api_motor.py."""
    saida.parent.mkdir(parents=True, exist_ok=True)

    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    feitos.add(json.loads(l)["abstract_id"])

    with saida.open("a", encoding="utf-8") as f:
        for abstract_id, texto in abstracts.items():
            if abstract_id in feitos:
                continue
            extracoes = extrair_com_prompt(client, system_prompt, texto)
            linha = {"abstract_id": abstract_id, "extracoes": extracoes}
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.2)


# ============================================================
# Normalizacao e matching (nucleo da funcao de avaliacao)
# ============================================================

_NUM_RE = re.compile(r"[-+]?\d[\d,\.]*")


def _extrair_numero(valor):
    """Extrai o primeiro numero de uma string tipo '971 ppm', '~1,000 ppm',
    '0.15%', devolvendo float ou None se nao achar nada."""
    if valor is None:
        return None
    s = str(valor).replace("·", ".")  # variante tipografica vista no corpus
    m = _NUM_RE.search(s)
    if not m:
        return None
    num_str = m.group(0).replace(",", "")
    try:
        return float(num_str)
    except ValueError:
        return None


def _valores_batem(v1, v2) -> bool:
    """Compara dois valores textuais pelo numero exato que carregam.
    Tolera apenas diferencas de FORMATACAO (espaco, virgula de milhar,
    til de aproximacao, ponto tipografico '·') -- nao tolera diferenca de
    digito. '971 ppm' e '972 ppm' sao considerados valores DIFERENTES,
    porque no dominio (valores citados diretamente do texto), uma
    diferenca de digito e um erro de leitura real, nao arredondamento."""
    n1, n2 = _extrair_numero(v1), _extrair_numero(v2)
    if n1 is None or n2 is None:
        return False
    return n1 == n2


def _extracao_bate(pred: dict, gold: dict) -> bool:
    """Define quando uma extracao PREDITA corresponde a uma extracao do
    GOLD, para fins de deteccao (independente de metric_type/entity_type
    estarem certos). Match e por valor numerico de 'value' e, quando
    is_range=True em ambos, tambem por 'value_max'."""
    if bool(pred.get("is_range")) != bool(gold.get("is_range")):
        return False
    if not _valores_batem(pred.get("value"), gold.get("value")):
        return False
    if gold.get("is_range"):
        if not _valores_batem(pred.get("value_max"), gold.get("value_max")):
            return False
    return True


def _casar_extracoes(preds: list, golds: list) -> list:
    """Casamento 1-para-1 (bipartite guloso) entre predicoes e gold de UM
    abstract. Devolve lista de tuplas (pred_idx_ou_None, gold_idx_ou_None):
      (i, j)     -> pred i bate com gold j (par verdadeiro-positivo)
      (i, None)  -> pred i nao bateu com nenhum gold (falso-positivo)
      (None, j)  -> gold j nao foi coberto por nenhuma pred (falso-negativo)
    """
    pares = []
    gold_usados = set()
    pred_usados = set()

    # passo 1: casar pares que batem exatamente por valor
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if j in gold_usados:
                continue
            if _extracao_bate(p, g):
                pares.append((i, j))
                gold_usados.add(j)
                pred_usados.add(i)
                break

    for i in range(len(preds)):
        if i not in pred_usados:
            pares.append((i, None))
    for j in range(len(golds)):
        if j not in gold_usados:
            pares.append((None, j))
    return pares


# ============================================================
# Metricas: precision / recall / F1, em duas camadas:
#   - deteccao: achou o valor certo (independente de categoria)
#   - classificacao: dado que achou, acertou metric_type / entity_type
# ============================================================

def avaliar(predicoes: dict, gold: dict) -> dict:
    """predicoes e gold: {abstract_id: [extracoes...]}. Devolve dict com
    metricas de deteccao e de classificacao, mais a lista de erros
    encontrados (para uso posterior como material do gradiente textual)."""
    tp = fp = fn = 0
    tp_metric_certo = tp_entity_certo = 0
    erros = []  # cada erro documentado para uso no ∇ do ProTeGi

    todos_ids = set(predicoes.keys()) | set(gold.keys())
    for abstract_id in todos_ids:
        preds = predicoes.get(abstract_id, [])
        golds = gold.get(abstract_id, [])
        pares = _casar_extracoes(preds, golds)

        for i, j in pares:
            if i is not None and j is not None:
                tp += 1
                p, g = preds[i], golds[j]
                metric_ok = p.get("metric_type") == g.get("metric_type")
                entity_ok = p.get("entity_type") == g.get("entity_type")
                if metric_ok:
                    tp_metric_certo += 1
                if entity_ok:
                    tp_entity_certo += 1
                if not metric_ok or not entity_ok:
                    erros.append({
                        "abstract_id": abstract_id,
                        "tipo_erro": "classificacao_incorreta",
                        "predito": p,
                        "esperado": g,
                    })
            elif i is not None and j is None:
                fp += 1
                erros.append({
                    "abstract_id": abstract_id,
                    "tipo_erro": "falso_positivo",
                    "predito": preds[i],
                    "esperado": None,
                })
            elif i is None and j is not None:
                fn += 1
                erros.append({
                    "abstract_id": abstract_id,
                    "tipo_erro": "falso_negativo",
                    "predito": None,
                    "esperado": golds[j],
                })

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    precisao_deteccao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_deteccao = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_deteccao = _f1(precisao_deteccao, recall_deteccao)

    acuracia_metric_type = tp_metric_certo / tp if tp > 0 else 0.0
    acuracia_entity_type = tp_entity_certo / tp if tp > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precisao_deteccao": round(precisao_deteccao, 4),
        "recall_deteccao": round(recall_deteccao, 4),
        "f1_deteccao": round(f1_deteccao, 4),
        "acuracia_metric_type_dado_match": round(acuracia_metric_type, 4),
        "acuracia_entity_type_dado_match": round(acuracia_entity_type, 4),
        "erros": erros,
    }


def dividir_gold(gold: dict, frac_treino: float = 0.7, seed: int = 42) -> tuple:
    """Divide o gold-standard em TREINO/TESTE de forma deterministica
    (mesma seed => mesmo split sempre, mesmo chamando essa funcao em
    processos separados como `otimizar` e `avaliar-prompt`).

    Isso existe para evitar vazamento entre a etapa de otimizacao do
    prompt (ProTeGi, que ve e otimiza F1 sobre o TREINO) e a avaliacao
    final do prompt (que deve rodar sobre o TESTE, nunca visto durante
    a otimizacao -- senao o F1 reportado fica enviesado/otimista).
    """
    ids = sorted(gold.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    corte = round(len(ids) * frac_treino)
    ids_treino = set(ids[:corte])
    gold_treino = {i: v for i, v in gold.items() if i in ids_treino}
    gold_teste = {i: v for i, v in gold.items() if i not in ids_treino}
    return gold_treino, gold_teste


def carregar_jsonl_por_abstract(caminho: Path) -> dict:
    """Le um .jsonl no formato {"abstract_id":..., "extracoes":[...]} e
    devolve {abstract_id: [extracoes...]}."""
    out = {}
    with caminho.open(encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            registro = json.loads(l)
            out[registro["abstract_id"]] = registro.get("extracoes", [])
    return out


# ============================================================
# ProTeGi: gradiente textual (∇) + edicao (δ) + selecao (TopK)
# ============================================================

PROMPT_GRADIENTE = """
I'm trying to write a zero-shot information extraction prompt for scientific abstracts about rare-earth elements (REE).

My current prompt is:
\"\"\"
{prompt}
\"\"\"

This prompt was tested against a small labeled dataset. Here are examples of MISTAKES it made, grouped by error type:

{erros_formatados}

Based on these mistakes, give {num_gradientes} distinct, concise reasons why the prompt could be producing these errors. Focus on missing or ambiguous CRITERIA in the prompt's category definitions — not on rephrasing. Each reason should point to a specific gap (e.g. "the prompt does not distinguish X from Y", "the prompt gives no rule for when a ratio/proportion should be Process_Metric vs Bulk_Concentration").

Wrap each reason with <START> and <END>.
""".strip()

PROMPT_EDICAO = """
I'm trying to write a zero-shot information extraction prompt for scientific abstracts about rare-earth elements (REE).

My current prompt is:
\"\"\"
{prompt}
\"\"\"

Based on testing, the problem with this prompt is: {gradiente}

Using this feedback, write {num_edicoes} improved versions of the prompt. Each version must:
- Keep the exact same output JSON field names, in the exact same order: raciocinio, value, value_max, is_range, sentence, metric_type, target_entity, entity_type, context_modifier
- Keep the instruction that "raciocinio" is a short free-text reasoning field, written BEFORE the other fields, explaining why the candidate was (or was not) considered valid and why this metric_type/entity_type was chosen
- Keep the exact same allowed values for metric_type (Bulk_Concentration, Individual_Entity_Concentration, Process_Metric, Invalid_Candidate) and entity_type (element, element_group, oxide, mineral, other)
- ADD or REFINE criteria/definitions to fix the identified problem — do not remove the JSON structure instructions, and do not introduce new fields or new category names.

Wrap each improved prompt with <START> and <END>.
""".strip()


def _formatar_erros_para_gradiente(erros: list, max_exemplos: int = 6) -> str:
    """Formata uma amostra de erros (priorizando diversidade de tipo) em
    texto legivel para o prompt de gradiente."""
    por_tipo = {}
    for e in erros:
        por_tipo.setdefault(e["tipo_erro"], []).append(e)

    linhas = []
    for tipo, lista in por_tipo.items():
        linhas.append(f"\n[{tipo}] ({len(lista)} casos, mostrando ate {max_exemplos}):")
        for e in lista[:max_exemplos]:
            if tipo == "falso_negativo":
                g = e["esperado"]
                linhas.append(
                    f"  - abstract {e['abstract_id']}: prompt MISSED a value that should "
                    f"have been extracted: value={g.get('value')!r} metric_type={g.get('metric_type')!r} "
                    f"target_entity={g.get('target_entity')!r} entity_type={g.get('entity_type')!r} "
                    f"(sentence: {g.get('sentence', '')[:200]!r})"
                )
            elif tipo == "falso_positivo":
                p = e["predito"]
                linhas.append(
                    f"  - abstract {e['abstract_id']}: prompt extracted a value that should NOT "
                    f"have been extracted: value={p.get('value')!r} metric_type={p.get('metric_type')!r} "
                    f"target_entity={p.get('target_entity')!r}"
                    + (f" | model's stated reasoning: {p.get('raciocinio')!r}" if p.get("raciocinio") else "")
                )
            else:  # classificacao_incorreta
                p, g = e["predito"], e["esperado"]
                linhas.append(
                    f"  - abstract {e['abstract_id']}: value={g.get('value')!r} was classified as "
                    f"metric_type={p.get('metric_type')!r}/entity_type={p.get('entity_type')!r} "
                    f"but should be metric_type={g.get('metric_type')!r}/entity_type={g.get('entity_type')!r} "
                    f"(target_entity: {g.get('target_entity')!r})"
                    + (f" | model's stated reasoning: {p.get('raciocinio')!r}" if p.get("raciocinio") else "")
                )
    return "\n".join(linhas)


def _extrair_blocos(texto: str) -> list:
    """Extrai todos os trechos entre <START> e <END>."""
    return re.findall(r"<START>(.*?)<END>", texto, re.DOTALL)


def gerar_gradientes(client: OpenAI, prompt_atual: str, erros: list,
                      num_gradientes: int = 3, max_erros_amostra: int = 12) -> list:
    amostra = erros[:max_erros_amostra]
    erros_fmt = _formatar_erros_para_gradiente(amostra)
    user_msg = PROMPT_GRADIENTE.format(
        prompt=prompt_atual, erros_formatados=erros_fmt, num_gradientes=num_gradientes
    )
    resposta = chamar_llm(client, "You are an expert prompt engineer.", user_msg,
                           temperature=TEMPERATURE_OTIMIZACAO, json_mode=False)
    gradientes = _extrair_blocos(resposta)
    return [g.strip() for g in gradientes if g.strip()]


def editar_prompt(client: OpenAI, prompt_atual: str, gradiente: str,
                   num_edicoes: int = 2) -> list:
    user_msg = PROMPT_EDICAO.format(
        prompt=prompt_atual, gradiente=gradiente, num_edicoes=num_edicoes
    )
    resposta = chamar_llm(client, "You are an expert prompt engineer.", user_msg,
                           temperature=TEMPERATURE_OTIMIZACAO, json_mode=False)
    candidatos = _extrair_blocos(resposta)
    return [c.strip() for c in candidatos if c.strip()]


def avaliar_prompt_no_minibatch(client: OpenAI, prompt_texto: str,
                                  minibatch_abstracts: dict, gold: dict) -> dict:
    """Roda o prompt sobre um minibatch de abstracts (em memoria, sem
    salvar em disco) e devolve o resultado de avaliar()."""
    predicoes = {}
    for abstract_id, texto in minibatch_abstracts.items():
        predicoes[abstract_id] = extrair_com_prompt(client, prompt_texto, texto)
    gold_minibatch = {k: v for k, v in gold.items() if k in minibatch_abstracts}
    return avaliar(predicoes, gold_minibatch)


def otimizar_protegi(client: OpenAI, abstracts: dict, gold: dict, saida_dir: Path,
                      passos: int = 4, beam_width: int = 3,
                      num_gradientes: int = 2, num_edicoes: int = 2,
                      tamanho_minibatch: int = 10) -> str:
    """Loop principal do ProTeGi (Algoritmo 1 do paper), simplificado:
      - beam de tamanho beam_width, iniciando com [PROMPT_APO_INICIAL]
      - a cada passo: expande cada prompt do beam (gradiente -> edicoes),
        avalia todos os candidatos no minibatch, mantem os beam_width
        melhores por F1 de deteccao (TopK greedy, sem bandit -- suficiente
        para o volume pequeno de dados deste projeto).
    Salva um log por passo em saida_dir e devolve o texto do prompt final.
    """
    saida_dir.mkdir(parents=True, exist_ok=True)
    ids_abstracts = list(abstracts.keys())

    beam = [PROMPT_APO_INICIAL]
    historico = []

    for passo in range(1, passos + 1):
        print(f"\n=== Passo {passo}/{passos} ===")
        # minibatch aleatorio (determinístico por passo, para reprodutibilidade simples)
        rng = random.Random(passo)
        amostra_ids = rng.sample(ids_abstracts, min(tamanho_minibatch, len(ids_abstracts)))
        minibatch = {i: abstracts[i] for i in amostra_ids}

        candidatos = list(beam)  # mantem os atuais tambem na disputa
        # guarda a avaliacao de cada prompt do beam pra nao recalcular na
        # selecao final (cada avaliacao ja custa `tamanho_minibatch` chamadas
        # de API, entao reavaliar o mesmo prompt no mesmo minibatch e desperdicio)
        resultados_beam = {}
        for prompt_atual in beam:
            resultado = avaliar_prompt_no_minibatch(client, prompt_atual, minibatch, gold)
            resultados_beam[prompt_atual] = resultado
            erros = resultado["erros"]
            if not erros:
                continue  # prompt ja perfeito no minibatch, nada a corrigir
            gradientes = gerar_gradientes(client, prompt_atual, erros, num_gradientes=num_gradientes)
            for g in gradientes:
                novos = editar_prompt(client, prompt_atual, g, num_edicoes=num_edicoes)
                candidatos.extend(novos)

        # avalia todos os candidatos no MESMO minibatch e seleciona TopK (greedy),
        # reaproveitando a avaliacao dos prompts do beam feita acima
        avaliados = []
        for c in candidatos:
            if c in resultados_beam:
                r = resultados_beam[c]
            else:
                r = avaliar_prompt_no_minibatch(client, c, minibatch, gold)
            avaliados.append((r["f1_deteccao"], c, r))

        avaliados.sort(key=lambda x: x[0], reverse=True)
        beam = [c for _, c, _ in avaliados[:beam_width]]

        melhor_f1, melhor_prompt, melhor_resultado = avaliados[0]
        print(f"  candidatos avaliados: {len(candidatos)} | melhor F1 (minibatch): {melhor_f1}")
        historico.append({
            "passo": passo,
            "melhor_f1_minibatch": melhor_f1,
            "tamanho_beam": len(beam),
            "num_candidatos_gerados": len(candidatos),
        })
        (saida_dir / f"passo_{passo}_melhor_prompt.txt").write_text(melhor_prompt, encoding="utf-8")

    (saida_dir / "historico.json").write_text(
        json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # prompt final: reavalia o beam inteiro no dataset COMPLETO (nao so o
    # ultimo minibatch) para escolher o vencedor final de forma mais robusta
    print("\n=== Selecao final: avaliando beam completo no dataset inteiro ===")
    resultados_finais = []
    for c in beam:
        r = avaliar(
            {i: extrair_com_prompt(client, c, t) for i, t in abstracts.items()},
            gold,
        )
        resultados_finais.append((r["f1_deteccao"], c, r))
    resultados_finais.sort(key=lambda x: x[0], reverse=True)
    melhor_f1_final, prompt_final, _ = resultados_finais[0]
    print(f"  F1 final (dataset completo) do prompt escolhido: {melhor_f1_final}")

    (saida_dir / "prompt_final.txt").write_text(prompt_final, encoding="utf-8")
    return prompt_final


# ============================================================
# CLI
# ============================================================

def cmd_avaliar_prompt(args):
    client = get_client()
    prompt_texto = Path(args.prompt).read_text(encoding="utf-8")
    abstracts = carregar_abstracts(Path(args.abstracts))
    gold = carregar_jsonl_por_abstract(Path(args.gold))

    if args.conjunto == "completo":
        gold_alvo = gold
    else:
        gold_treino, gold_teste = dividir_gold(gold, frac_treino=args.frac_treino, seed=args.seed)
        gold_alvo = gold_treino if args.conjunto == "treino" else gold_teste

    abstracts_alvo = {i: t for i, t in abstracts.items() if i in gold_alvo}
    print(f"Rodando prompt {args.prompt} sobre {len(abstracts_alvo)} abstracts "
          f"(conjunto={args.conjunto}, seed={args.seed}, frac_treino={args.frac_treino})...")
    if args.conjunto != "completo":
        print("  (esse split e o mesmo usado em `otimizar` com o mesmo --seed/--frac-treino, "
              "entao 'teste' aqui e sempre o holdout nao visto pelo ProTeGi.)")
    rodar_prompt_sobre_abstracts(client, prompt_texto, abstracts_alvo, Path(args.output))
    predicoes = carregar_jsonl_por_abstract(Path(args.output))
    resultado = avaliar(predicoes, gold_alvo)
    _imprimir_resultado(Path(args.prompt).stem, resultado)


def cmd_otimizar(args):
    client = get_client()
    abstracts = carregar_abstracts(Path(args.abstracts))
    gold = carregar_jsonl_por_abstract(Path(args.gold))

    gold_treino, gold_teste = dividir_gold(gold, frac_treino=args.frac_treino, seed=args.seed)
    abstracts_treino = {i: t for i, t in abstracts.items() if i in gold_treino}
    print(f"Split treino/teste: {len(gold_treino)} treino / {len(gold_teste)} teste "
          f"(de {len(gold)} no gold-standard, seed={args.seed}, frac_treino={args.frac_treino}).")
    print(f"Iniciando ProTeGi com {len(abstracts_treino)} abstracts de TREINO, "
          f"{args.passos} passos, beam={args.beam}...")

    prompt_final = otimizar_protegi(
        client, abstracts_treino, gold_treino, Path(args.output),
        passos=args.passos, beam_width=args.beam,
    )

    saida_dir = Path(args.output)
    (saida_dir / "holdout_teste_ids.json").write_text(
        json.dumps(sorted(gold_teste.keys()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nPrompt final salvo em:", saida_dir / "prompt_final.txt")
    print(f"IDs de TESTE (holdout, nao usados na otimizacao) salvos em: "
          f"{saida_dir / 'holdout_teste_ids.json'}")
    print("Para avaliar sem vazamento, rode `avaliar-prompt` com o MESMO --seed/--frac-treino "
          "(o padrao --conjunto teste ja faz isso automaticamente).")


def cmd_extrair_tudo(args):
    """Roda o prompt vencedor sobre TODOS os abstracts do corpus (nao so os
    que tem gold-standard) e monta uma amostra estratificada por metric_type
    para auditoria manual via o campo 'raciocinio' -- valida sem depender de
    metricas supervisionadas, ja que nao ha rotulo pra a maioria dos casos."""
    client = get_client()
    prompt_texto = Path(args.prompt).read_text(encoding="utf-8")
    abstracts = carregar_abstracts(Path(args.abstracts))
    print(f"Extraindo com {args.prompt} sobre TODOS os {len(abstracts)} abstracts do corpus...")
    rodar_prompt_sobre_abstracts(client, prompt_texto, abstracts, Path(args.output))
    predicoes = carregar_jsonl_por_abstract(Path(args.output))

    ids_para_amostra = set(predicoes.keys())
    if args.gold:
        gold_ids = set(carregar_jsonl_por_abstract(Path(args.gold)).keys())
        excluidos = gold_ids & ids_para_amostra
        ids_para_amostra -= gold_ids
        print(f"  (excluindo {len(excluidos)} abstracts que ja tem gold-standard da amostra de auditoria "
              f"-- esses ja sao validados via avaliar-prompt/F1)")

    pool_por_estrato = {}
    for abstract_id in sorted(ids_para_amostra):
        for extracao in predicoes.get(abstract_id, []):
            metric_type = extracao.get("metric_type", "desconhecido")
            item = dict(extracao)
            item["abstract_id"] = abstract_id
            pool_por_estrato.setdefault(metric_type, []).append(item)

    rng = random.Random(args.seed)
    amostra = []
    print("\nComposicao da amostra de auditoria por metric_type:")
    for metric_type, itens in sorted(pool_por_estrato.items()):
        n = min(args.por_estrato, len(itens))
        selecionados = rng.sample(itens, n)
        amostra.extend(selecionados)
        print(f"  {metric_type}: {n} amostrados de {len(itens)} disponiveis")

    amostra_path = Path(args.amostra_auditoria)
    amostra_path.parent.mkdir(parents=True, exist_ok=True)
    with amostra_path.open("w", encoding="utf-8") as f:
        for item in amostra:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nExtracao completa salva em: {args.output} ({len(predicoes)} abstracts, "
          f"{sum(len(v) for v in predicoes.values())} extracoes)")
    print(f"Amostra de auditoria ({len(amostra)} extracoes, {len(pool_por_estrato)} metric_types) "
          f"salva em: {amostra_path}")
    print("Revise cada linha lendo 'sentence' + 'raciocinio' contra o abstract original.")


def cmd_comparar(args):
    gold = carregar_jsonl_por_abstract(Path(args.gold))
    manual = carregar_jsonl_por_abstract(Path(args.manual))
    apo = carregar_jsonl_por_abstract(Path(args.apo))
    r_manual = avaliar(manual, gold)
    r_apo = avaliar(apo, gold)
    _imprimir_resultado("MANUAL", r_manual)
    _imprimir_resultado("APO", r_apo)
    print("\n=== COMPARACAO ===")
    print(f"{'Metrica':<35}{'Manual':>10}{'APO':>10}")
    for chave, rotulo in [
        ("precisao_deteccao", "Precisao (deteccao)"),
        ("recall_deteccao", "Recall (deteccao)"),
        ("f1_deteccao", "F1 (deteccao)"),
        ("acuracia_metric_type_dado_match", "Acuracia metric_type"),
        ("acuracia_entity_type_dado_match", "Acuracia entity_type"),
    ]:
        print(f"{rotulo:<35}{r_manual[chave]:>10}{r_apo[chave]:>10}")


def _imprimir_resultado(nome: str, resultado: dict):
    print(f"\n--- Resultado: {nome} ---")
    print(f"  TP={resultado['tp']}  FP={resultado['fp']}  FN={resultado['fn']}")
    print(f"  Precisao (deteccao): {resultado['precisao_deteccao']}")
    print(f"  Recall (deteccao):   {resultado['recall_deteccao']}")
    print(f"  F1 (deteccao):       {resultado['f1_deteccao']}")
    print(f"  Acuracia metric_type (dado match): {resultado['acuracia_metric_type_dado_match']}")
    print(f"  Acuracia entity_type (dado match): {resultado['acuracia_entity_type_dado_match']}")


def main():
    parser = argparse.ArgumentParser(description="APO/ProTeGi para extracao de REE.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p2 = sub.add_parser("avaliar-prompt", help="Roda um prompt arbitrario (arquivo .txt) no gold-standard e avalia.")
    p2.add_argument("--prompt", required=True)
    p2.add_argument("--gold", required=True)
    p2.add_argument("--abstracts", required=True)
    p2.add_argument("--output", required=True)
    p2.add_argument("--conjunto", choices=["treino", "teste", "completo"], default="teste",
                     help="Qual fatia do gold-standard avaliar. 'teste' (padrao) e o holdout "
                          "nunca visto pelo ProTeGi durante `otimizar` -- use para medir "
                          "generalizacao sem vazamento. Use --seed/--frac-treino iguais aos "
                          "usados em `otimizar` para reproduzir o mesmo split.")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--frac-treino", type=float, default=0.7, dest="frac_treino")
    p2.set_defaults(func=cmd_avaliar_prompt)

    p3 = sub.add_parser("otimizar", help="Roda o ciclo ProTeGi a partir do p0 ingenuo.")
    p3.add_argument("--gold", required=True)
    p3.add_argument("--abstracts", required=True)
    p3.add_argument("--output", required=True, help="Diretorio para salvar logs e prompt final.")
    p3.add_argument("--passos", type=int, default=4)
    p3.add_argument("--beam", type=int, default=3)
    p3.add_argument("--seed", type=int, default=42,
                     help="Seed do split treino/teste (70/30) do gold-standard.")
    p3.add_argument("--frac-treino", type=float, default=0.7, dest="frac_treino",
                     help="Fracao do gold-standard usada como treino (ProTeGi so ve isso).")
    p3.set_defaults(func=cmd_otimizar)

    p3b = sub.add_parser("extrair-tudo", help="Roda o prompt final sobre TODO o corpus (nao so o gold) e monta amostra estratificada para auditoria manual.")
    p3b.add_argument("--prompt", required=True, help="Caminho para prompt_final.txt (ou outro prompt .txt).")
    p3b.add_argument("--abstracts", required=True, help="CSV com o corpus completo (ex: os 58 pos-filtro).")
    p3b.add_argument("--output", required=True, help="Arquivo .jsonl com a extracao de TODOS os abstracts.")
    p3b.add_argument("--amostra-auditoria", required=True, dest="amostra_auditoria",
                      help="Arquivo .jsonl com a amostra estratificada por metric_type, para revisao manual.")
    p3b.add_argument("--por-estrato", type=int, default=5, dest="por_estrato",
                      help="Quantas extracoes amostrar por metric_type (default 5).")
    p3b.add_argument("--gold", default=None,
                      help="Opcional: gold-standard, para EXCLUIR da amostra os abstracts que ja tem rotulo "
                           "(esses ja sao validados via avaliar-prompt).")
    p3b.add_argument("--seed", type=int, default=42)
    p3b.set_defaults(func=cmd_extrair_tudo)

    p4 = sub.add_parser("comparar", help="Compara predicoes MANUAL vs APO contra o gold-standard.")
    p4.add_argument("--manual", required=True)
    p4.add_argument("--apo", required=True)
    p4.add_argument("--gold", required=True)
    p4.set_defaults(func=cmd_comparar)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()