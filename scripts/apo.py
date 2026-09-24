"""Otimizacao automatica de prompt (APO / ProTeGi) para extracao de REE.

Implementa o ciclo de "gradiente textual" descrito em Pryzant et al. 2023
(Automatic Prompt Optimization with "Gradient Descent" and Beam Search)
aplicado a extracao estruturada de valores quantitativos de REE, otimizando
um prompt simples/ingenuo (p0) a partir dos erros que ele comete contra o
gold-standard anotado manualmente.

Fontes de dados (casadas pelo campo "doc_id"):

  - --gold (.jsonl): as 119 extracoes anotadas manualmente (o gold-standard
    em si), SEM o texto do abstract:
      {"doc_id": "...", "extracoes": [{"value":..., "metric_type":..., ...}],
       "particao": "dev"}
    O campo "particao" (valores observados: "dev", "val", "test") E o
    split usado pelo script -- ver PARTICAO_PARA_CONJUNTO. Nao ha split
    recalculado por seed: a divisao treino/validacao/teste e sempre a que
    ja vem da anotacao.
  - --candidatos (.jsonl): o corpus completo (362 resumos), COM o texto:
      {"doc_id": "...", "titulo": "...", "resumo": "..."}
    Fonte PRIMARIA de texto para qualquer doc_id, inclusive os do gold.
  - --resumos-extra (.csv, opcional): ex. TERRAS_RARAS.csv, com colunas
    'id_openalex' (URL tipo "https://openalex.org/W4200261466") e
    'abstract'. Fonte SECUNDARIA de texto, usada so para doc_id do gold
    que NAO aparecem em --candidatos -- controle-negativo (resumos sem
    nada pra extrair, por isso nunca entraram no corpus de candidatos).

O split treino/validacao/teste vem do campo "particao" de --gold
(dev->treino, val->validacao, test->teste; ver PARTICAO_PARA_CONJUNTO):

  - `otimizar` roda o ProTeGi sobre o TREINO; ao final de cada passo do
    beam, o prompt vencedor do passo tambem e avaliado na VALIDACAO
    inteira (so para log/plot, nunca influencia a selecao do beam); a
    selecao FINAL do prompt vencedor (ao fim de todos os passos) tambem
    usa a VALIDACAO, nunca o TESTE.
  - `avaliar-prompt` (--conjunto teste, padrao) avalia sobre o TESTE --
    holdout nunca visto em nenhuma etapa da otimizacao -- para nao
    inflar o F1 reportado. --conjunto validacao/treino/completo tambem
    estao disponiveis para diagnostico.

Uso tipico:

  # 1. Rodar o ciclo de otimizacao ProTeGi a partir do p0 ingenuo (so no treino):
  python apo.py otimizar --gold gold.jsonl --candidatos candidatos.jsonl --resumos-extra TERRAS_RARAS.csv --output apo_run/

  # 2. Avaliar o prompt final produzido pelo APO no holdout de teste:
  python apo.py avaliar-prompt --prompt apo_run/prompt_final.txt --gold gold.jsonl --candidatos candidatos.jsonl --resumos-extra TERRAS_RARAS.csv --output apo_predicoes_teste.jsonl

  # 3. Rodar o prompt final sobre o corpus completo de candidatos:
  python apo.py extrair-tudo --prompt apo_run/prompt_final.txt --candidatos candidatos.jsonl --output extracao_completa.jsonl --amostra-auditoria amostra_auditoria.jsonl

python apo.py otimizar --gold ../gold_standard/gold_final.jsonl --candidatos ../data/candidatos.jsonl --resumos-extra ../data/TERRAS_RARAS.csv --output ../outputs/output_apo_oficial --passos 5 --beam 3

python apo.py avaliar-prompt --prompt ../outputs/output_apo_oficial/prompt_final.txt --gold ../gold_standard/gold_final.jsonl --candidatos ../data/candidatos.jsonl --output ../outputs/output_apo_oficial/predicoes_teste.jsonl --conjunto teste

python apo.py extrair-tudo --prompt ../outputs/output_apo_oficial/prompt_final.txt --candidatos ../data/candidatos.jsonl --resumos-extra ../data/TERRAS_RARAS.csv --output ../outputs/output_apo_oficial/extracao_completa.jsonl --amostra-auditoria ../outputs/output_apo_oficial/amostra_auditoria.jsonl --gold ../gold_standard/gold_final.jsonl

"""

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_URL = "https://iluma.cnpem.br:4000/v1"
CONFIG_PADRAO = Path(__file__).resolve().parent / "config.yaml"

# Temperatura da etapa de otimizacao (gradiente/edicao do ProTeGi). Nao vem
# do YAML: o YAML controla a EXTRACAO; aqui queremos diversidade de propostas.
# Passa sempre pelo piso da instalacao (ver chamar_llm).
TEMPERATURE_OTIMIZACAO = 1.0

# Configuracao do LLM. Preenchida por carregar_config() no inicio de main();
# os valores abaixo sao so o fallback caso o modulo seja importado sem YAML.
# Credenciais NUNCA ficam aqui nem no YAML: vem de chave/chave.env.
LLM_CFG = {
    "modelo": "iluma",
    "temperatura_piso": 0.5,
    "temperatura": 0.5,
    "max_tokens": 16000,
    "max_tokens_teto": 16000,
    "timeout_s": 120,
    "tentativas": 3,
    "k_autoconsistencia": 1,
}

# Seed fixa para TODO o script (amostragem de minibatch, amostragem da
# auditoria em `extrair-tudo`). Nao e exposta via CLI de proposito.
SEED = 42

# O split treino/validacao/teste usa a particao JA ATRIBUIDA na anotacao
# (campo "particao" de --gold), nao um split recalculado por seed. Mapeia
# os valores observados no corpus para os tres conjuntos do script.
PARTICAO_PARA_CONJUNTO = {"dev": "treino", "val": "validacao", "test": "teste"}

# ============================================================
# Prompt inicial (p0 ingenuo): estrutura de OUTPUT fixa (nomes de campo,
# categorias permitidas -- alinhadas ao guia de anotacao, secoes 3.4/3.6/
# 3.8/3.9), mas SEM nenhuma definicao do que cada categoria significa e
# sem few-shot. Isso e o que o ciclo ProTeGi deve preencher sozinho.
# ============================================================

PROMPT_APO_INICIAL = """
You extract quantitative data about rare earth elements from scientific abstracts.

Read the abstract below and return every numeric value you find that relates to
rare earth elements.

Answer with JSON only, using this exact format:

{"extracoes": [
  {
    "value": 0.3,
    "value_max": null,
    "is_range": false,
    "unit": "wt%",
    "sentence": "the sentence where the value appears",
    "metric_type": "Bulk Concentration",
    "target_entity": "TREO",
    "entity_type": "oxide",
    "context_modifier": "none"
  }
]}

Allowed values:

- metric_type: "Bulk Concentration", "Individual Entity Concentration",
  "Process Metric", "Invalid Candidate"
- entity_type: "element", "element_group", "oxide", "mineral", "other"
- context_modifier: "none", "average", "approximate", "greater_than",
  "less_than", "up_to"

If the abstract has no relevant values, answer {"extracoes": []}.
""".strip()


# ============================================================
# Configuracao (config.yaml) e cliente API
# ============================================================

def carregar_config(caminho: Path = CONFIG_PADRAO) -> dict:
    """Le a secao `llm` de config.yaml, valida e grava em LLM_CFG.

    Invariantes checados aqui (falha cedo, antes de gastar chamadas):
      - temperatura >= temperatura_piso (a instalacao rejeita abaixo do piso);
      - max_tokens <= max_tokens_teto;
      - tentativas >= 1 e k_autoconsistencia >= 1 (inteiros).
    """
    global LLM_CFG
    if not caminho.exists():
        raise FileNotFoundError(f"config.yaml nao encontrado em {caminho}.")
    with caminho.open(encoding="utf-8") as f:
        bruto = yaml.safe_load(f) or {}
    if "llm" not in bruto:
        raise ValueError(f"{caminho}: secao 'llm' ausente.")
    cfg = {**LLM_CFG, **bruto["llm"]}

    desconhecidas = set(bruto["llm"]) - set(LLM_CFG)
    if desconhecidas:
        raise ValueError(f"{caminho}: chaves desconhecidas em 'llm': {sorted(desconhecidas)}")

    if cfg["temperatura"] < cfg["temperatura_piso"]:
        raise ValueError(
            f"temperatura ({cfg['temperatura']}) abaixo do piso da instalacao "
            f"({cfg['temperatura_piso']}).")
    if cfg["max_tokens"] > cfg["max_tokens_teto"]:
        raise ValueError(
            f"max_tokens ({cfg['max_tokens']}) acima do teto ({cfg['max_tokens_teto']}).")
    for chave in ("tentativas", "k_autoconsistencia", "max_tokens", "timeout_s"):
        if not isinstance(cfg[chave], int) or cfg[chave] < 1:
            raise ValueError(f"'{chave}' deve ser inteiro >= 1 (recebido: {cfg[chave]!r}).")

    LLM_CFG = cfg
    return cfg


def get_client() -> OpenAI:
    load_dotenv(BASE_DIR / "chave" / "chave.env")
    token = os.getenv("ILUMA_API_KEY")
    if not token:
        raise RuntimeError("ILUMA_API_KEY não encontrada em chave.env.")
    # `tentativas` no YAML = numero TOTAL de tentativas; o SDK conta so as
    # retentativas apos a primeira, entao max_retries = tentativas - 1.
    return OpenAI(
        base_url=BASE_URL,
        api_key=token,
        timeout=float(LLM_CFG["timeout_s"]),
        max_retries=LLM_CFG["tentativas"] - 1,
    )


def chamar_llm(client: OpenAI, system_prompt: str, user_content: str,
                temperature: float = None, json_mode: bool = True,
                habilitar_thinking: bool = False) -> str:
    """Chamada generica ao LLM. Retorna o texto da resposta (string).

    Temperatura: se `temperature` for None usa LLM_CFG["temperatura"]; em
    qualquer caso o valor efetivo e max(temperatura, temperatura_piso),
    porque esta instalacao rejeita valores abaixo do piso.

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
    if temperature is None:
        temperature = LLM_CFG["temperatura"]
    temperature = max(temperature, LLM_CFG["temperatura_piso"])

    kwargs = dict(
        model=LLM_CFG["modelo"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=LLM_CFG["max_tokens"],
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
                "raciocinio ou aumentar max_tokens em config.yaml, respeitando "
                "max_tokens_teto)"
            )
        raise RuntimeError(f"A API não retornou conteúdo ({detalhes}).")
    return response.choices[0].message.content


# ============================================================
# Carregamento de dados (.jsonl, casados por doc_id)
# ============================================================

def carregar_gold(caminho_jsonl: Path) -> dict:
    """Le --gold (doc_id, extracoes, particao) -- o gold-standard em si,
    SEM o texto do abstract -- e devolve
    {doc_id: {"extracoes": [...], "particao": "dev"/"val"/"test"}}.

    O split treino/validacao/teste usado por este script e sempre a
    particao ja atribuida na anotacao (ver PARTICAO_PARA_CONJUNTO e
    dividir_gold_por_particao), nao um split recalculado por seed.
    """
    gold = {}
    with caminho_jsonl.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            registro = json.loads(linha)
            gold[registro["doc_id"]] = {
                "extracoes": registro.get("extracoes", []),
                "particao": registro.get("particao"),
            }
    return gold


def dividir_gold_por_particao(gold: dict) -> tuple:
    """Divide o gold-standard em TREINO / VALIDACAO / TESTE usando o campo
    'particao' ja atribuido na anotacao (dev->treino, val->validacao,
    test->teste, ver PARTICAO_PARA_CONJUNTO), em vez de recalcular um
    split por seed. Devolve tres dicts {doc_id: [extracoes...]}, no
    formato que avaliar() espera.

    - TREINO: unico conjunto que o ProTeGi ve/otimiza (gradiente + edicao).
    - VALIDACAO: avaliada ao final de cada passo do beam e usada para a
      selecao final do prompt vencedor -- nunca usada para gerar gradiente.
    - TESTE: holdout nunca visto durante a otimizacao, usado so na
      avaliacao final (`avaliar-prompt --conjunto teste`), para nao
      inflar o F1 reportado.

    Levanta erro se algum doc_id tiver 'particao' ausente ou um valor fora
    de PARTICAO_PARA_CONJUNTO, para pegar cedo qualquer valor novo/errado
    que o mapeamento ainda nao cubra.
    """
    grupos = {"treino": {}, "validacao": {}, "teste": {}}
    desconhecidas = set()
    for doc_id, item in gold.items():
        conjunto = PARTICAO_PARA_CONJUNTO.get(item.get("particao"))
        if conjunto is None:
            desconhecidas.add(item.get("particao"))
            continue
        grupos[conjunto][doc_id] = item["extracoes"]
    if desconhecidas:
        raise ValueError(
            f"Valor(es) de 'particao' sem mapeamento conhecido: {sorted(desconhecidas, key=str)}. "
            f"Mapeamento atual: {PARTICAO_PARA_CONJUNTO}. Ajuste PARTICAO_PARA_CONJUNTO no script "
            f"para cobrir o(s) valor(es) novo(s)."
        )
    return grupos["treino"], grupos["validacao"], grupos["teste"]


def carregar_candidatos(caminho_jsonl: Path) -> dict:
    """Le --candidatos (doc_id, titulo, resumo), com o texto do abstract, e
    devolve {doc_id: resumo}. E a fonte PRIMARIA de texto usada pelo
    script, inclusive para os doc_id que tambem aparecem em --gold."""
    out = {}
    with caminho_jsonl.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            registro = json.loads(linha)
            out[registro["doc_id"]] = registro.get("resumo", "")
    return out


def carregar_resumos_csv(caminho_csv: Path) -> dict:
    """Le um CSV com pelo menos as colunas 'id_openalex' (URL tipo
    'https://openalex.org/W4200261466') e 'abstract', e devolve
    {doc_id: abstract} extraindo o doc_id do final da URL.

    Usado como fonte de texto SECUNDARIA (--resumos-extra), para doc_id do
    gold que nao aparecem em --candidatos: controle-negativo, resumos que
    nao tinham nada para extrair e por isso nunca entraram no corpus de
    candidatos."""
    out = {}
    with caminho_csv.open(encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            url = (linha.get("id_openalex") or "").strip()
            if not url:
                continue
            doc_id = url.rstrip("/").rsplit("/", 1)[-1]
            out[doc_id] = linha.get("abstract", "")
    return out


def carregar_ids_jsonl(caminho: Path) -> set:
    """Le qualquer .jsonl com campo 'doc_id' e devolve so o conjunto de ids
    (usado para excluir da amostra de auditoria os docs que ja tem gold)."""
    ids = set()
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            ids.add(json.loads(linha)["doc_id"])
    return ids


def textos_para_ids(ids, candidatos: dict, resumos_extra: dict = None) -> dict:
    """Busca o texto de cada doc_id em `ids` (tipicamente um subconjunto do
    gold-standard, apos o split): primeiro em `candidatos` (--candidatos),
    e para quem nao aparecer la, em `resumos_extra` (--resumos-extra,
    opcional, ex. TERRAS_RARAS.csv). Avisa e ignora qualquer doc_id que
    nao apareca em nenhuma das duas fontes, ja que sem texto o abstract
    nao pode ser extraido."""
    resumos_extra = resumos_extra or {}
    out = {}
    faltando = []
    usados_extra = 0
    for i in ids:
        if i in candidatos:
            out[i] = candidatos[i]
        elif i in resumos_extra:
            out[i] = resumos_extra[i]
            usados_extra += 1
        else:
            faltando.append(i)
    if usados_extra:
        print(f"  [nota] {usados_extra} doc_id(s) do gold nao estavam em --candidatos e "
              f"tiveram o texto buscado em --resumos-extra.")
    if faltando:
        print(f"  [aviso] {len(faltando)} doc_id(s) do gold sem texto em --candidatos nem "
              f"--resumos-extra, ignorados: {sorted(faltando)[:5]}"
              f"{' ...' if len(faltando) > 5 else ''}")
    return out


# ============================================================
# Extracao: roda um prompt (p0 ou candidato do APO) sobre um
# conjunto de abstracts.
# ============================================================

def _chave_voto(extracao: dict) -> tuple:
    """Identidade de uma extracao para fins de votacao: os campos que
    definem O QUE foi extraido (valor, faixa, unidade, entidade-alvo).
    Campos de classificacao (metric_type etc.) ficam de fora de proposito:
    duas amostras que extraem o mesmo valor mas classificam diferente
    votam juntas, e a classificacao e decidida por maioria dentro do grupo."""
    def norm(x):
        return str(x).strip().lower() if x is not None else None
    return (norm(extracao.get("value")), norm(extracao.get("value_max")),
            norm(extracao.get("unit")), norm(extracao.get("target_entity")))


def _votar_extracoes(amostras: list) -> list:
    """Votacao por autoconsistencia sobre k listas de extracoes.

    Votar a resposta inteira quase nunca daria maioria (listas de objetos
    raramente coincidem por completo), entao o voto e por EXTRACAO: uma
    extracao sobrevive se aparece em pelo menos ceil(k/2) amostras. Dentro
    de cada grupo sobrevivente, cada campo restante (metric_type,
    entity_type, context_modifier, ...) recebe o valor mais frequente; o
    desempate vai para a primeira amostra em que a extracao apareceu.
    """
    k = len(amostras)
    minimo = (k + 1) // 2
    grupos = {}
    for lista in amostras:
        vistos_nesta_amostra = set()
        for ex in lista:
            if not isinstance(ex, dict):
                continue
            chave = _chave_voto(ex)
            if chave in vistos_nesta_amostra:
                continue  # a mesma amostra nao vota duas vezes na mesma extracao
            vistos_nesta_amostra.add(chave)
            grupos.setdefault(chave, []).append(ex)

    resultado = []
    for chave, membros in grupos.items():
        if len(membros) < minimo:
            continue
        consolidada = dict(membros[0])
        campos = set().union(*(m.keys() for m in membros))
        for campo in campos:
            valores = [json.dumps(m.get(campo), sort_keys=True, ensure_ascii=False)
                       for m in membros if campo in m]
            mais_comum = Counter(valores).most_common(1)[0][0]
            consolidada[campo] = json.loads(mais_comum)
        resultado.append(consolidada)
    return resultado


def _extrair_uma_amostra(client, system_prompt, abstract_texto):
    """Uma chamada + parse. Devolve a lista de extracoes, ou None se falhou."""
    import time
    t = time.time()
    try:
        raw = chamar_llm(client, system_prompt, abstract_texto)
        print(f"    [ok em {time.time()-t:.1f}s | prompt={len(system_prompt)} chars | abstract={len(abstract_texto)} chars]")
        data = json.loads(raw)
        extracoes = data.get("extracoes", [])
        return extracoes if isinstance(extracoes, list) else None
    except Exception as e:
        print(f"    [erro em {time.time()-t:.1f}s | prompt={len(system_prompt)} chars | abstract={len(abstract_texto)} chars | {type(e).__name__}: {e}]")
        return None


def extrair_com_prompt(client: OpenAI, system_prompt: str, abstract_texto: str) -> list:
    """Roda o prompt (system_prompt) sobre um abstract e devolve a lista de
    extracoes (ja parseada), tolerando erro de parsing (devolve lista vazia
    e nao derruba o restante do lote).

    Com k_autoconsistencia == 1 (congelado, ver config.yaml) e exatamente
    uma chamada, igual ao comportamento anterior. Com k > 1, faz k chamadas
    e consolida por votacao (ver _votar_extracoes); amostras que falharam
    nao votam, mas o quorum continua calculado sobre as k pedidas."""
    k = LLM_CFG["k_autoconsistencia"]
    if k == 1:
        return _extrair_uma_amostra(client, system_prompt, abstract_texto) or []

    amostras = [_extrair_uma_amostra(client, system_prompt, abstract_texto) for _ in range(k)]
    validas = [a for a in amostras if a is not None]
    if not validas:
        return []
    return _votar_extracoes(validas)


def rodar_prompt_sobre_abstracts(client: OpenAI, system_prompt: str,
                                   abstracts: dict, saida: Path) -> None:
    """Roda system_prompt sobre cada abstract em `abstracts` (dict
    doc_id->texto) e grava incrementalmente em `saida` (.jsonl, uma linha
    por doc_id), de forma retomavel (pula doc_id ja presentes em `saida`)."""
    saida.parent.mkdir(parents=True, exist_ok=True)

    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    feitos.add(json.loads(l)["doc_id"])

    with saida.open("a", encoding="utf-8") as f:
        for doc_id, texto in abstracts.items():
            if doc_id in feitos:
                continue
            extracoes = extrair_com_prompt(client, system_prompt, texto)
            linha = {"doc_id": doc_id, "extracoes": extracoes}
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.2)


def carregar_jsonl_por_doc(caminho: Path) -> dict:
    """Le um .jsonl no formato {"doc_id":..., "extracoes":[...]} (saida de
    rodar_prompt_sobre_abstracts) e devolve {doc_id: [extracoes...]}."""
    out = {}
    with caminho.open(encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            registro = json.loads(l)
            out[registro["doc_id"]] = registro.get("extracoes", [])
    return out


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
    """predicoes e gold: {doc_id: [extracoes...]}. Devolve dict com
    metricas de deteccao e de classificacao, mais a lista de erros
    encontrados (para uso posterior como material do gradiente textual)."""
    tp = fp = fn = 0
    tp_metric_certo = tp_entity_certo = 0
    erros = []  # cada erro documentado para uso no ∇ do ProTeGi

    todos_ids = set(predicoes.keys()) | set(gold.keys())
    for doc_id in todos_ids:
        preds = predicoes.get(doc_id, [])
        golds = gold.get(doc_id, [])
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
                        "doc_id": doc_id,
                        "tipo_erro": "classificacao_incorreta",
                        "predito": p,
                        "esperado": g,
                    })
            elif i is not None and j is None:
                fp += 1
                erros.append({
                    "doc_id": doc_id,
                    "tipo_erro": "falso_positivo",
                    "predito": preds[i],
                    "esperado": None,
                })
            elif i is None and j is not None:
                fn += 1
                erros.append({
                    "doc_id": doc_id,
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


def _resumo_metricas(r: dict) -> dict:
    """Extrai de um resultado de avaliar() so os campos numericos (sem a
    lista 'erros', que e grande e nao serve para plot), para log granular
    no historico.json."""
    return {
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
        "precisao_deteccao": r["precisao_deteccao"],
        "recall_deteccao": r["recall_deteccao"],
        "f1_deteccao": r["f1_deteccao"],
        "acuracia_metric_type_dado_match": r["acuracia_metric_type_dado_match"],
        "acuracia_entity_type_dado_match": r["acuracia_entity_type_dado_match"],
    }


def _imprimir_resultado(nome: str, resultado: dict):
    print(f"\n--- Resultado: {nome} ---")
    print(f"  TP={resultado['tp']}  FP={resultado['fp']}  FN={resultado['fn']}")
    print(f"  Precisao (deteccao): {resultado['precisao_deteccao']}")
    print(f"  Recall (deteccao):   {resultado['recall_deteccao']}")
    print(f"  F1 (deteccao):       {resultado['f1_deteccao']}")
    print(f"  Acuracia metric_type (dado match): {resultado['acuracia_metric_type_dado_match']}")
    print(f"  Acuracia entity_type (dado match): {resultado['acuracia_entity_type_dado_match']}")


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

Based on these mistakes, give {num_gradientes} distinct, concise reasons why the prompt could be producing these errors. Focus on missing or ambiguous CRITERIA in the prompt's category definitions — not on rephrasing. Each reason should point to a specific gap (e.g. "the prompt does not distinguish X from Y", "the prompt gives no rule for when a ratio/proportion should be Process Metric vs Bulk Concentration").

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
- Keep the exact same output JSON field names, in the exact same order: raciocinio, value, value_max, is_range, unit, sentence, metric_type, target_entity, entity_type, context_modifier
- Keep the instruction that "raciocinio" is a short free-text reasoning field, written BEFORE the other fields, explaining why the candidate was (or was not) considered valid and why this metric_type/entity_type was chosen
- Keep the exact same allowed values for metric_type ("Bulk Concentration", "Individual Entity Concentration", "Process Metric", "Invalid Candidate") and entity_type ("element", "element_group", "oxide", "mineral", "other")
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
                    f"  - doc {e['doc_id']}: prompt MISSED a value that should "
                    f"have been extracted: value={g.get('value')!r} metric_type={g.get('metric_type')!r} "
                    f"target_entity={g.get('target_entity')!r} entity_type={g.get('entity_type')!r} "
                    f"(sentence: {g.get('sentence', '')[:200]!r})"
                )
            elif tipo == "falso_positivo":
                p = e["predito"]
                linhas.append(
                    f"  - doc {e['doc_id']}: prompt extracted a value that should NOT "
                    f"have been extracted: value={p.get('value')!r} metric_type={p.get('metric_type')!r} "
                    f"target_entity={p.get('target_entity')!r}"
                    + (f" | model's stated reasoning: {p.get('raciocinio')!r}" if p.get("raciocinio") else "")
                )
            else:  # classificacao_incorreta
                p, g = e["predito"], e["esperado"]
                linhas.append(
                    f"  - doc {e['doc_id']}: value={g.get('value')!r} was classified as "
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


def avaliar_prompt_no_conjunto(client: OpenAI, prompt_texto: str,
                                 abstracts_subset: dict, gold_subset: dict) -> dict:
    """Roda o prompt sobre um conjunto de abstracts (em memoria, sem salvar
    em disco) e devolve o resultado de avaliar(). Usado tanto para o
    minibatch de treino quanto para a validacao inteira."""
    predicoes = {}
    for doc_id, texto in abstracts_subset.items():
        predicoes[doc_id] = extrair_com_prompt(client, prompt_texto, texto)
    return avaliar(predicoes, gold_subset)


def _carregar_checkpoint(saida_dir: Path):
    """Procura, em `saida_dir`, o checkpoint de uma rodada anterior (e
    possivelmente interrompida) de `otimizar`: `historico.json` casado com
    o `beam_apos_passo_N.json` do ultimo passo registrado, mais a
    trajetoria (`candidatos.json`) se existir.

    Se `beam_apos_passo_N.json` nao existir (rodada de uma versao anterior
    a essa funcionalidade, que so salvava o VENCEDOR de cada passo em
    `passo_N_melhor_prompt.txt`, nao o beam inteiro), cai para um resumo
    DEGRADADO: reconstroi um beam de tamanho 1 a partir desse vencedor, e
    ainda assim retoma dali -- perde a diversidade dos outros prompts do
    beam antigo E a linhagem (pais/passo de criacao) desse prompt na
    trajetoria, mas reaproveita o progresso principal (nao refaz os
    passos ja feitos do zero).

    Devolve um dict {historico, beam, ultimo_passo, degradado, trajetoria,
    chamadas_metrica}, ou None se nao houver nada pra retomar (sem
    historico.json, ou historico.json vazio). Levanta RuntimeError se
    achar historico.json com passos registrados mas nem o beam nem o
    vencedor do ultimo passo existirem -- para nao arriscar sobrescrever
    esse historico silenciosamente."""
    historico_path = saida_dir / "historico.json"
    if not historico_path.exists():
        return None
    historico = json.loads(historico_path.read_text(encoding="utf-8"))
    if not historico:
        return None
    ultimo_passo = historico[-1]["passo"]

    trajetoria, chamadas_metrica = [], 0
    candidatos_path = saida_dir / "candidatos.json"
    if candidatos_path.exists():
        dados = json.loads(candidatos_path.read_text(encoding="utf-8"))
        trajetoria = dados.get("candidatas", [])
        chamadas_metrica = dados.get("chamadas_metrica", 0)

    beam_path = saida_dir / f"beam_apos_passo_{ultimo_passo}.json"
    if beam_path.exists():
        beam = json.loads(beam_path.read_text(encoding="utf-8"))
        return {"historico": historico, "beam": beam, "ultimo_passo": ultimo_passo,
                "degradado": False, "trajetoria": trajetoria, "chamadas_metrica": chamadas_metrica}

    melhor_path = saida_dir / f"passo_{ultimo_passo}_melhor_prompt.txt"
    if melhor_path.exists():
        print(f"  [aviso] {historico_path.name} tem {ultimo_passo} passo(s), mas "
              f"{beam_path.name} nao existe (checkpoint de uma versao anterior a "
              f"esta funcionalidade) -- retomando em modo DEGRADADO: o beam vai "
              f"reiniciar so com o vencedor salvo em {melhor_path.name}, perdendo "
              f"a diversidade dos outros prompts do beam antigo E a linhagem desse "
              f"prompt na trajetoria (vai aparecer como um no novo, sem pais), mas "
              f"sem refazer os {ultimo_passo} passo(s) ja feitos.")
        beam = [melhor_path.read_text(encoding="utf-8")]
        return {"historico": historico, "beam": beam, "ultimo_passo": ultimo_passo,
                "degradado": True, "trajetoria": trajetoria, "chamadas_metrica": chamadas_metrica}

    print(f"  [aviso] {historico_path.name} tem {ultimo_passo} passo(s), mas nem "
          f"{beam_path.name} nem {melhor_path.name} existem -- nao ha nada pra "
          f"retomar. Para nao arriscar sobrescrever esse historico antigo, pare "
          f"e mova/renomeie {saida_dir} antes de rodar de novo, ou apague-o de "
          f"proposito se realmente quiser comecar do zero.")
    raise RuntimeError(
        f"Checkpoint inconsistente em {saida_dir}: historico.json existe mas nao "
        f"ha beam nem prompt vencedor pra retomar (nem para um resumo degradado)."
    )


def otimizar_protegi(client: OpenAI,
                      abstracts_treino: dict, gold_treino: dict,
                      abstracts_val: dict, gold_val: dict,
                      saida_dir: Path,
                      passos: int = 4, beam_width: int = 3,
                      num_gradientes: int = 2, num_edicoes: int = 2,
                      tamanho_minibatch: int = 10, retomar: bool = True,
                      nome_inicial: str = "p0_ingenuo") -> str:
    """Loop principal do ProTeGi (Algoritmo 1 do paper), simplificado:
      - beam de tamanho beam_width, iniciando com [PROMPT_APO_INICIAL]
      - a cada passo: expande cada prompt do beam (gradiente -> edicoes)
        usando os erros no minibatch de TREINO, avalia todos os
        candidatos nesse mesmo minibatch e mantem os beam_width melhores
        por F1 de deteccao (TopK greedy, sem bandit -- suficiente para o
        volume pequeno de dados deste projeto).

    `tamanho_minibatch=None` usa o TREINO INTEIRO a cada passo; passe um
    inteiro menor que len(abstracts_treino) para minibatches parciais.

    Alem do `historico.json` (log por passo, ja existente), salva
    `candidatos.json` com a TRAJETORIA completa da busca, no formato
    esperado por scripts de analise externos (ex. `analisar.py`/GEPA):

        {"inicial": nome_inicial, "melhor": <indice>,
         "chamadas_metrica": <int>,
         "candidatas": [{"indice", "pais", "passo", "no_beam",
                          "nota_val", "nota_treino", "chamadas_ate_aqui",
                          "instrucoes"}, ...]}

    Cada prompt DISTINTO vira um no, registrado so na PRIMEIRA vez que
    aparece (a raiz, indice 0, e o PROMPT_APO_INICIAL, com pais=[None] e
    passo=0). "pais" e o indice do prompt do qual ele foi gerado via
    gradiente+edicao. "nota_treino"/"nota_val" sao o F1 de deteccao desse
    prompt no minibatch do seu passo de criacao e na VALIDACAO inteira --
    validacao agora e calculada para TODO candidato novo (nao so o
    vencedor de cada passo), o que aumenta o custo de API do `otimizar`
    (mais `len(abstracts_val)` chamadas por candidato novo), mas e o que
    permite plotar a trajetoria completa depois sem rodar de novo.
    "chamadas_ate_aqui" e o total de chamadas de EXTRACAO (nao conta
    gradiente/edicao, que so geram texto de prompt) gastas ANTES de
    avaliar esse candidato. "no_beam" reflete a selecao MAIS RECENTE em
    que o candidato participou (sobreviventes de passos anteriores tem
    esse campo atualizado a cada passo em que continuam disputando, mas
    NAO ganham um novo no -- o no e unico por texto de prompt).

    A selecao final (apos o ultimo passo) NAO reavalia o beam -- escolhe,
    entre os prompts que sobraram no beam, o de maior "nota_val" ja
    registrada na trajetoria (que e exatamente a mesma avaliacao que uma
    reavaliacao repetiria, ja que o conjunto de validacao nunca muda).

    Retomada (retomar=True, padrao): se `saida_dir` ja tiver um checkpoint
    de uma rodada anterior (mesmo interrompida no meio), continua a partir
    do ultimo passo concluido -- incluindo a trajetoria e o contador de
    chamadas ja acumulados -- em vez de gastar API reavaliando os passos
    ja feitos (ver `_carregar_checkpoint`). Passe retomar=False (ou
    `--reiniciar` na CLI) para ignorar qualquer checkpoint e comecar do
    PROMPT_APO_INICIAL, sobrescrevendo os arquivos de `saida_dir`.
    """
    saida_dir.mkdir(parents=True, exist_ok=True)
    ids_treino = list(abstracts_treino.keys())
    tamanho_mb = tamanho_minibatch if tamanho_minibatch else len(ids_treino)

    beam = [PROMPT_APO_INICIAL]
    historico = []
    trajetoria = []
    indice_por_prompt = {}
    chamadas_metrica = 0
    passo_inicial = 1
    if retomar:
        ck = _carregar_checkpoint(saida_dir)
        if ck is not None:
            historico, beam = ck["historico"], ck["beam"]
            trajetoria, chamadas_metrica = ck["trajetoria"], ck["chamadas_metrica"]
            indice_por_prompt = {no["instrucoes"]: no["indice"] for no in trajetoria}
            passo_inicial = ck["ultimo_passo"] + 1
            modo = " (modo DEGRADADO, beam reduzido a 1 prompt)" if ck["degradado"] else ""
            print(f"  [retomando] checkpoint encontrado em {saida_dir} com "
                  f"{ck['ultimo_passo']} passo(s) ja feito(s), {len(trajetoria)} "
                  f"no(s) na trajetoria, {chamadas_metrica} chamadas de metrica ja "
                  f"gastas -- continuando do passo {passo_inicial}{modo}.")
    if passo_inicial > passos:
        print(f"  [retomando] checkpoint ja tem {passo_inicial - 1} passo(s), >= "
              f"--passos={passos} pedido agora; pulando direto para a selecao final "
              f"com o beam salvo.")

    def _gravar_candidatos_json(melhor_indice=None):
        (saida_dir / "candidatos.json").write_text(json.dumps(
            {"inicial": nome_inicial, "melhor": melhor_indice,
             "chamadas_metrica": chamadas_metrica, "candidatas": trajetoria},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")

    for passo in range(passo_inicial, passos + 1):
        print(f"\n=== Passo {passo}/{passos} ===")
        # minibatch de treino, deterministico por passo (seed fixa + passo)
        rng = random.Random(SEED + passo)
        amostra_ids = rng.sample(ids_treino, min(tamanho_mb, len(ids_treino)))
        minibatch = {i: abstracts_treino[i] for i in amostra_ids}
        gold_minibatch = {i: gold_treino[i] for i in amostra_ids}

        candidatos = list(beam)  # mantem os atuais tambem na disputa
        pai_de = {}  # texto do candidato NOVO -> texto do prompt que o gerou (so este passo)
        # guarda a avaliacao de cada prompt do beam pra nao recalcular na
        # selecao (cada avaliacao ja custa `len(minibatch)` chamadas de API)
        resultados_beam = {}
        for prompt_atual in beam:
            resultado = avaliar_prompt_no_conjunto(client, prompt_atual, minibatch, gold_minibatch)
            chamadas_metrica += len(minibatch)
            resultados_beam[prompt_atual] = resultado
            erros = resultado["erros"]
            if not erros:
                continue  # prompt ja perfeito no minibatch, nada a corrigir
            gradientes = gerar_gradientes(client, prompt_atual, erros, num_gradientes=num_gradientes)
            for g in gradientes:
                novos = editar_prompt(client, prompt_atual, g, num_edicoes=num_edicoes)
                for n in novos:
                    pai_de.setdefault(n, prompt_atual)
                candidatos.extend(novos)

        # avalia todos os candidatos no MESMO minibatch e seleciona TopK,
        # reaproveitando a avaliacao dos prompts do beam feita acima
        avaliados = []
        for c in candidatos:
            if c in resultados_beam:
                r = resultados_beam[c]
            else:
                r = avaliar_prompt_no_conjunto(client, c, minibatch, gold_minibatch)
                chamadas_metrica += len(minibatch)
            avaliados.append((r["f1_deteccao"], c, r))

            # registra o no na trajetoria na PRIMEIRA vez que esse texto de
            # prompt aparece (candidatos que ja sao nos existentes -- ex.
            # sobreviventes do beam -- nao geram um no novo)
            if c not in indice_por_prompt:
                if c == PROMPT_APO_INICIAL:
                    pais_indices, passo_criacao = [None], 0
                else:
                    pai_texto = pai_de.get(c)
                    pai_indice = indice_por_prompt.get(pai_texto)
                    pais_indices, passo_criacao = [pai_indice], passo
                chamadas_antes = chamadas_metrica
                r_val = avaliar_prompt_no_conjunto(client, c, abstracts_val, gold_val)
                chamadas_metrica += len(abstracts_val)
                indice = len(trajetoria)
                trajetoria.append({
                    "indice": indice, "pais": pais_indices, "passo": passo_criacao,
                    "no_beam": False,
                    "nota_val": r_val["f1_deteccao"], "nota_treino": r["f1_deteccao"],
                    "chamadas_ate_aqui": chamadas_antes, "instrucoes": c,
                })
                indice_por_prompt[c] = indice

        avaliados.sort(key=lambda x: x[0], reverse=True)
        beam = [c for _, c, _ in avaliados[:beam_width]]

        # "no_beam" reflete a selecao deste passo pra TODO candidato
        # avaliado aqui, inclusive sobreviventes de passos anteriores
        beam_textos = set(beam)
        for _, c, _ in avaliados:
            trajetoria[indice_por_prompt[c]]["no_beam"] = c in beam_textos

        melhor_f1, melhor_prompt, melhor_resultado = avaliados[0]
        print(f"  candidatos avaliados: {len(candidatos)} | melhor F1 (treino/minibatch): {melhor_f1}")
        print(f"  chamadas de metrica acumuladas: {chamadas_metrica}")

        historico.append({
            "passo": passo,
            "tamanho_minibatch_treino": len(minibatch),
            "num_candidatos_gerados": len(candidatos),
            "tamanho_beam": len(beam),
            "chamadas_metrica_ate_aqui": chamadas_metrica,
            # metricas de TODO candidato avaliado neste passo (para plot de
            # dispersao/evolucao da populacao inteira, nao so do vencedor)
            "candidatos_treino": [
                {"prompt_preview": c[:120], **_resumo_metricas(r)}
                for _, c, r in avaliados
            ],
            "beam_apos_selecao_f1_treino": [f1 for f1, _, _ in avaliados[:beam_width]],
            "melhor_treino": _resumo_metricas(melhor_resultado),
        })
        (saida_dir / f"passo_{passo}_melhor_prompt.txt").write_text(melhor_prompt, encoding="utf-8")
        # beam completo (nao so o vencedor) -- e o que permite retomar do
        # ponto exato onde parou, em vez de so do melhor prompt do passo
        (saida_dir / f"beam_apos_passo_{passo}.json").write_text(
            json.dumps(beam, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # regrava a cada passo (nao so no final), pra nao perder o log
        # granular se o processo for interrompido no meio de uma rodada longa
        (saida_dir / "historico.json").write_text(
            json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _gravar_candidatos_json(melhor_indice=None)  # "melhor" so e conhecido no final

    # selecao final: escolhe, dentro do beam final, quem tem a MAIOR
    # nota_val ja registrada na trajetoria -- nao precisa reavaliar, pois
    # e exatamente a mesma avaliacao (mesmo conjunto de validacao) que foi
    # feita quando o candidato foi criado
    indices_do_beam_final = [indice_por_prompt[c] for c in beam]
    indice_melhor = max(indices_do_beam_final, key=lambda i: trajetoria[i]["nota_val"])
    prompt_final = trajetoria[indice_melhor]["instrucoes"]
    print(f"\n=== Selecao final: maior nota_val no beam = "
          f"{trajetoria[indice_melhor]['nota_val']} (indice {indice_melhor}) ===")

    (saida_dir / "prompt_final.txt").write_text(prompt_final, encoding="utf-8")
    _gravar_candidatos_json(melhor_indice=indice_melhor)
    return prompt_final


# ============================================================
# CLI
# ============================================================

def cmd_avaliar_prompt(args):
    client = get_client()
    prompt_texto = Path(args.prompt).read_text(encoding="utf-8")
    gold = carregar_gold(Path(args.gold))
    candidatos = carregar_candidatos(Path(args.candidatos))
    resumos_extra = carregar_resumos_csv(Path(args.resumos_extra)) if args.resumos_extra else {}

    gold_treino, gold_val, gold_teste = dividir_gold_por_particao(gold)
    if args.conjunto == "completo":
        gold_alvo = {**gold_treino, **gold_val, **gold_teste}
    else:
        gold_alvo = {"treino": gold_treino, "validacao": gold_val, "teste": gold_teste}[args.conjunto]

    abstracts_alvo = textos_para_ids(gold_alvo.keys(), candidatos, resumos_extra)
    print(f"Rodando prompt {args.prompt} sobre {len(abstracts_alvo)} abstracts "
          f"(conjunto={args.conjunto}, particao do gold: dev=treino/val=validacao/test=teste)...")
    if args.conjunto != "completo":
        print("  (esse split vem do campo 'particao' de --gold, o mesmo usado em `otimizar`, "
              "entao 'teste' aqui e sempre o holdout nao visto pelo ProTeGi.)")
    rodar_prompt_sobre_abstracts(client, prompt_texto, abstracts_alvo, Path(args.output))
    predicoes = carregar_jsonl_por_doc(Path(args.output))
    resultado = avaliar(predicoes, gold_alvo)
    _imprimir_resultado(Path(args.prompt).stem, resultado)


def cmd_otimizar(args):
    client = get_client()
    gold = carregar_gold(Path(args.gold))
    candidatos = carregar_candidatos(Path(args.candidatos))
    resumos_extra = carregar_resumos_csv(Path(args.resumos_extra)) if args.resumos_extra else {}

    gold_treino, gold_val, gold_teste = dividir_gold_por_particao(gold)
    abstracts_treino = textos_para_ids(gold_treino.keys(), candidatos, resumos_extra)
    abstracts_val = textos_para_ids(gold_val.keys(), candidatos, resumos_extra)
    extracoes_treino = gold_treino
    extracoes_val = gold_val

    print(f"Split treino/validacao/teste (particao dev/val/test de --gold): "
          f"{len(gold_treino)}/{len(gold_val)}/{len(gold_teste)} (de {len(gold)} no gold-standard).")
    print(f"Iniciando ProTeGi com {len(abstracts_treino)} abstracts de TREINO, "
          f"{args.passos} passos, beam={args.beam}, "
          f"minibatch={args.tamanho_minibatch or 'treino inteiro'}...")

    prompt_final = otimizar_protegi(
        client, abstracts_treino, extracoes_treino, abstracts_val, extracoes_val,
        Path(args.output), passos=args.passos, beam_width=args.beam,
        tamanho_minibatch=args.tamanho_minibatch, retomar=not args.reiniciar,
        nome_inicial=args.nome_inicial,
    )

    saida_dir = Path(args.output)
    (saida_dir / "holdout_teste_ids.json").write_text(
        json.dumps(sorted(gold_teste.keys()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nPrompt final salvo em:", saida_dir / "prompt_final.txt")
    print(f"Trajetoria completa (formato GEPA/analisar.py) salva em: {saida_dir / 'candidatos.json'}")
    print(f"IDs de TESTE (holdout, nunca visto na otimizacao) salvos em: "
          f"{saida_dir / 'holdout_teste_ids.json'}")
    print("Para avaliar sem vazamento, rode `avaliar-prompt` com os MESMOS --gold/--candidatos "
          "(o split vem sempre do campo 'particao' de --gold; o padrao --conjunto teste "
          "ja usa o holdout automaticamente).")


def cmd_extrair_tudo(args):
    """Roda o prompt vencedor sobre TODOS os abstracts de --candidatos (nao
    so os que tem gold-standard) e monta uma amostra estratificada por
    metric_type para auditoria manual via o campo 'raciocinio' -- valida
    sem depender de metricas supervisionadas, ja que nao ha rotulo pra a
    maioria dos casos."""
    client = get_client()
    prompt_texto = Path(args.prompt).read_text(encoding="utf-8")
    abstracts = carregar_candidatos(Path(args.candidatos))
    print(f"Extraindo com {args.prompt} sobre TODOS os {len(abstracts)} abstracts do corpus (candidatos)...")
    rodar_prompt_sobre_abstracts(client, prompt_texto, abstracts, Path(args.output))
    predicoes = carregar_jsonl_por_doc(Path(args.output))

    ids_para_amostra = set(predicoes.keys())
    if args.gold:
        gold_ids = carregar_ids_jsonl(Path(args.gold))
        excluidos = gold_ids & ids_para_amostra
        ids_para_amostra -= gold_ids
        print(f"  (excluindo {len(excluidos)} abstracts que ja tem gold-standard da amostra de auditoria "
              f"-- esses ja sao validados via avaliar-prompt/F1)")

    pool_por_estrato = {}
    for doc_id in sorted(ids_para_amostra):
        for extracao in predicoes.get(doc_id, []):
            metric_type = extracao.get("metric_type", "desconhecido")
            item = dict(extracao)
            item["doc_id"] = doc_id
            pool_por_estrato.setdefault(metric_type, []).append(item)

    rng = random.Random(SEED)
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


def main():
    parser = argparse.ArgumentParser(description="APO/ProTeGi para extracao de REE.")
    sub = parser.add_subparsers(dest="comando", required=True)

    def _add_arg_config(p):
        p.add_argument("--config", default=str(CONFIG_PADRAO),
                        help="Caminho do config.yaml (padrao: config.yaml ao lado do script).")

    def _add_arg_resumos_extra(p):
        p.add_argument("--resumos-extra", default=None, dest="resumos_extra",
                        help="Opcional: CSV com colunas 'id_openalex' (URL) e 'abstract' "
                             "(ex. TERRAS_RARAS.csv), usado como fonte de texto SECUNDARIA "
                             "para doc_id do gold que nao aparecem em --candidatos "
                             "(controle-negativo: resumos sem nada pra extrair).")

    p2 = sub.add_parser("avaliar-prompt", help="Roda um prompt arbitrario (arquivo .txt) no gold-standard e avalia.")
    p2.add_argument("--prompt", required=True)
    p2.add_argument("--gold", required=True, help="Extracoes anotadas manualmente (doc_id, extracoes, particao), sem texto.")
    p2.add_argument("--candidatos", required=True, help="Corpus completo (doc_id, titulo, resumo) -- fonte primaria do texto dos abstracts do gold.")
    p2.add_argument("--output", required=True)
    p2.add_argument("--conjunto", choices=["treino", "validacao", "teste", "completo"], default="teste",
                     help="Qual fatia do gold-standard avaliar. 'teste' (padrao) e o holdout "
                          "nunca visto pelo ProTeGi durante `otimizar` -- use para medir "
                          "generalizacao sem vazamento. O split vem do campo 'particao' de "
                          "--gold (dev->treino, val->validacao, test->teste), o mesmo usado "
                          "em `otimizar`.")
    _add_arg_resumos_extra(p2)
    _add_arg_config(p2)
    p2.set_defaults(func=cmd_avaliar_prompt)

    p3 = sub.add_parser("otimizar", help="Roda o ciclo ProTeGi a partir do p0 ingenuo.")
    p3.add_argument("--gold", required=True, help="Extracoes anotadas manualmente (doc_id, extracoes, particao), sem texto.")
    p3.add_argument("--candidatos", required=True, help="Corpus completo (doc_id, titulo, resumo) -- fonte primaria do texto dos abstracts do gold.")
    p3.add_argument("--output", required=True, help="Diretorio para salvar logs e prompt final.")
    p3.add_argument("--passos", type=int, default=4)
    p3.add_argument("--beam", type=int, default=3)
    p3.add_argument("--tamanho-minibatch", type=int, default=None, dest="tamanho_minibatch",
                     help="Tamanho do minibatch de TREINO usado a cada passo. Padrao: treino inteiro.")
    p3.add_argument("--reiniciar", action="store_true",
                     help="Ignora qualquer checkpoint existente em --output e comeca do zero "
                          "(PROMPT_APO_INICIAL), sobrescrevendo os arquivos la. Sem essa flag, "
                          "se --output ja tiver historico.json + beam_apos_passo_N.json de uma "
                          "rodada anterior (mesmo interrompida), a otimizacao retoma do ultimo "
                          "passo concluido em vez de refazer chamadas de API ja pagas.")
    p3.add_argument("--nome-inicial", default="p0_ingenuo", dest="nome_inicial",
                     help="Rotulo do prompt inicial gravado no campo 'inicial' de candidatos.json "
                          "(ex. 'v00_ingenuo'). So identifica a rodada nos plots -- nao muda o "
                          "prompt em si.")
    _add_arg_resumos_extra(p3)
    _add_arg_config(p3)
    p3.set_defaults(func=cmd_otimizar)

    p3b = sub.add_parser("extrair-tudo", help="Roda o prompt final sobre TODO o corpus de candidatos e monta amostra estratificada para auditoria manual.")
    p3b.add_argument("--prompt", required=True, help="Caminho para prompt_final.txt (ou outro prompt .txt).")
    p3b.add_argument("--candidatos", required=True, help="Arquivo .jsonl com o corpus completo (doc_id, titulo, resumo), sem rotulo.")
    p3b.add_argument("--output", required=True, help="Arquivo .jsonl com a extracao de TODOS os abstracts.")
    p3b.add_argument("--amostra-auditoria", required=True, dest="amostra_auditoria",
                      help="Arquivo .jsonl com a amostra estratificada por metric_type, para revisao manual.")
    p3b.add_argument("--por-estrato", type=int, default=5, dest="por_estrato",
                      help="Quantas extracoes amostrar por metric_type (default 5).")
    p3b.add_argument("--gold", default=None,
                      help="Opcional: arquivo de gold-standard (doc_id, extracoes, ...), usado so para EXCLUIR "
                           "da amostra os doc_id que ja tem gold (esses ja sao validados via avaliar-prompt).")
    _add_arg_config(p3b)
    p3b.set_defaults(func=cmd_extrair_tudo)

    args = parser.parse_args()
    cfg = carregar_config(Path(args.config))
    print(f"[config] modelo={cfg['modelo']} temperatura={cfg['temperatura']} "
          f"(piso {cfg['temperatura_piso']}) max_tokens={cfg['max_tokens']} "
          f"timeout={cfg['timeout_s']}s tentativas={cfg['tentativas']} "
          f"k={cfg['k_autoconsistencia']}")
    args.func(args)


if __name__ == "__main__":
    main()