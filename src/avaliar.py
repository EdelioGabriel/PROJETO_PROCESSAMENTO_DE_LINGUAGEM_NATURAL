"""Harness de avaliacao (F3-01 a F3-04).

Compara um conjunto de extracoes previstas com o gold, documento a documento.

Duas decisoes de projeto, tomadas explicitamente:

**Alinhamento guloso** (F3-01). Um documento produz varias extracoes e a ordem
nao e garantida, entao comparar posicionalmente esta errado. O alinhamento
guloso pontua todos os pares possiveis, ordena e casa do melhor para o pior.
Escolhido em vez do hungaro por ser mais simples de explicar no relatorio; com
poucas extracoes por documento, os dois quase sempre produzem o mesmo
resultado.

**Tolerancia relativa de 1%** (F3-02). "0.30" e "0.3" sao o mesmo valor, e a
conversao de unidades introduz arredondamento. Sem tolerancia, isso contaria
como erro.

Dois niveis de metrica, porque medem coisas diferentes:

- **deteccao**: o par foi casado, ou seja, o valor foi encontrado. Mede
  cobertura, e ignora se os campos categoricos estao certos.
- **tupla completa**: o par foi casado E todos os campos batem. E a metrica
  primaria do projeto.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from src.entidades import Lexico
from src.schema import Extracao, Registro

TOLERANCIA = 0.01

CAMPOS_TUPLA = (
    "value",
    "value_max",
    "is_range",
    "unit",
    "metric_type",
    "target_entity",
    "entity_type",
    "context_modifier",
)
CAMPOS_CATEGORICOS = ("metric_type", "entity_type", "context_modifier", "unit")


# ---------------------------------------------------------------------
# Comparacao de campos
# ---------------------------------------------------------------------


def valores_casam(a: float | None, b: float | None, tol: float = TOLERANCIA) -> bool:
    """Igualdade numerica com tolerancia relativa."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    escala = max(abs(a), abs(b))
    return abs(a - b) <= tol * escala


def normalizar_entidade(texto: str) -> str:
    """Compara `target_entity` sem caixa, pontuacao ou plural trivial."""
    texto = re.sub(r"[^a-z0-9]+", "", texto.lower())
    return texto[:-1] if texto.endswith("s") and len(texto) > 2 else texto


_LEXICO: Lexico | None = None


def mesma_entidade(a: str, b: str) -> bool:
    """Duas grafias da mesma entidade contam como iguais.

    Primeiro a comparacao textual (caixa, pontuacao e plural ignorados); se
    falhar, as duas sao levadas ao nome canonico do lexico, de modo que
    "rare-earth elements", "REEs" e "REE" casam, assim como "REE 2 O 3" e
    "TREO". Entidades fora do lexico continuam exigindo a mesma grafia.
    """
    global _LEXICO
    if normalizar_entidade(a) == normalizar_entidade(b):
        return True
    if _LEXICO is None:
        _LEXICO = Lexico.carregar()
    canonico_a = _LEXICO.canonico(a)
    return canonico_a is not None and canonico_a == _LEXICO.canonico(b)


def campos_iguais(campo: str, ouro: Extracao, previsto: Extracao) -> bool:
    a, b = getattr(ouro, campo), getattr(previsto, campo)
    if campo in ("value", "value_max"):
        return valores_casam(a, b)
    if campo == "target_entity":
        return mesma_entidade(a, b)
    return a == b


def pontuar_par(ouro: Extracao, previsto: Extracao) -> float | None:
    """Similaridade de um par candidato, ou None se o par e impossivel.

    O `value` e a chave: sem ele casando, nao ha par. Os demais campos so
    desempatam quando duas previsoes disputam o mesmo ouro.
    """
    if not valores_casam(ouro.value, previsto.value):
        return None
    pontos = 1.0
    if valores_casam(ouro.value_max, previsto.value_max):
        pontos += 0.5
    if ouro.unit == previsto.unit:
        pontos += 0.5
    if mesma_entidade(ouro.target_entity, previsto.target_entity):
        pontos += 0.5
    return pontos


# ---------------------------------------------------------------------
# Alinhamento
# ---------------------------------------------------------------------


@dataclass
class Alinhamento:
    pares: list[tuple[Extracao, Extracao]] = field(default_factory=list)
    nao_encontradas: list[Extracao] = field(default_factory=list)  # falsos negativos
    inventadas: list[Extracao] = field(default_factory=list)  # falsos positivos


def alinhar(ouro: list[Extracao], previsto: list[Extracao]) -> Alinhamento:
    """Casa extracoes previstas com as do gold, do melhor par para o pior."""
    candidatos = []
    for i, o in enumerate(ouro):
        for j, p in enumerate(previsto):
            pontos = pontuar_par(o, p)
            if pontos is not None:
                candidatos.append((pontos, i, j))
    candidatos.sort(key=lambda c: (-c[0], c[1], c[2]))

    usados_ouro: set[int] = set()
    usados_previsto: set[int] = set()
    resultado = Alinhamento()
    for _, i, j in candidatos:
        if i in usados_ouro or j in usados_previsto:
            continue
        usados_ouro.add(i)
        usados_previsto.add(j)
        resultado.pares.append((ouro[i], previsto[j]))

    resultado.nao_encontradas = [o for i, o in enumerate(ouro) if i not in usados_ouro]
    resultado.inventadas = [p for j, p in enumerate(previsto) if j not in usados_previsto]
    return resultado


def tupla_completa(ouro: Extracao, previsto: Extracao) -> bool:
    return all(campos_iguais(c, ouro, previsto) for c in CAMPOS_TUPLA)


# ---------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precisao": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def contagens_por_documento(
    ouro: dict[str, Registro], previsto: dict[str, Registro]
) -> dict[str, dict]:
    """Contagens brutas por documento. O bootstrap reamostra estas unidades."""
    saida = {}
    for doc_id, reg_ouro in ouro.items():
        reg_prev = previsto.get(doc_id, Registro(doc_id=doc_id))
        a = alinhar(reg_ouro.extracoes, reg_prev.extracoes)
        completos = sum(1 for o, p in a.pares if tupla_completa(o, p))
        saida[doc_id] = {
            "deteccao_tp": len(a.pares),
            "deteccao_fp": len(a.inventadas),
            "deteccao_fn": len(a.nao_encontradas),
            "tupla_tp": completos,
            "tupla_fp": len(a.inventadas) + (len(a.pares) - completos),
            "tupla_fn": len(a.nao_encontradas) + (len(a.pares) - completos),
            "pares": a.pares,
        }
    return saida


def agregar(contagens: dict[str, dict], prefixo: str) -> dict:
    return prf(
        sum(c[f"{prefixo}_tp"] for c in contagens.values()),
        sum(c[f"{prefixo}_fp"] for c in contagens.values()),
        sum(c[f"{prefixo}_fn"] for c in contagens.values()),
    )


def acuracia_por_campo(contagens: dict[str, dict]) -> dict[str, dict]:
    """Acerto de cada campo entre os pares casados.

    Restrito aos pares: um campo so pode ser julgado quando ha o que comparar.
    A cobertura aparece separadamente, nas metricas de deteccao.
    """
    saida = {}
    for campo in CAMPOS_TUPLA:
        certos = total = 0
        for c in contagens.values():
            for o, p in c["pares"]:
                total += 1
                certos += campos_iguais(campo, o, p)
        saida[campo] = {
            "acertos": certos,
            "total": total,
            "acuracia": certos / total if total else 0.0,
        }
    return saida


def f1_macro(contagens: dict[str, dict], campo: str) -> dict:
    """F1 macro de um campo categorico, sobre os pares casados."""
    pares = [
        (getattr(o, campo), getattr(p, campo))
        for c in contagens.values()
        for o, p in c["pares"]
    ]
    rotulos = sorted({r for par in pares for r in par}, key=str)
    por_rotulo = {}
    for rotulo in rotulos:
        tp = sum(1 for a, b in pares if a == rotulo and b == rotulo)
        fp = sum(1 for a, b in pares if a != rotulo and b == rotulo)
        fn = sum(1 for a, b in pares if a == rotulo and b != rotulo)
        por_rotulo[rotulo] = prf(tp, fp, fn)
    macro = sum(v["f1"] for v in por_rotulo.values()) / len(por_rotulo) if por_rotulo else 0.0
    return {"macro_f1": macro, "por_rotulo": por_rotulo}


def matriz_confusao(contagens: dict[str, dict], campo: str) -> Counter:
    return Counter(
        (getattr(o, campo), getattr(p, campo))
        for c in contagens.values()
        for o, p in c["pares"]
    )


def taxa_alucinacao(previsto: dict[str, Registro], resumos: dict[str, str]) -> dict:
    """Fracao de extracoes cuja `sentence` nao e trecho literal do resumo.

    Extensao do verificador de ancoragem (F4-03) para a avaliacao: uma extracao
    bem formada que cita texto inexistente e invencao, nao erro de campo.
    """
    total = ancoradas = 0
    for doc_id, reg in previsto.items():
        resumo = re.sub(r"\s+", " ", resumos.get(doc_id, ""))
        for e in reg.extracoes:
            total += 1
            ancoradas += re.sub(r"\s+", " ", e.sentence) in resumo
    return {
        "total": total,
        "ancoradas": ancoradas,
        "taxa": 1 - ancoradas / total if total else 0.0,
    }


# ---------------------------------------------------------------------
# Estatistica
# ---------------------------------------------------------------------


def bootstrap_ic(
    contagens: dict[str, dict],
    prefixo: str = "tupla",
    n: int = 1000,
    nivel: float = 0.95,
    semente: int = 42,
) -> tuple[float, float]:
    """Intervalo de confianca por reamostragem DE DOCUMENTOS.

    Reamostrar extracoes violaria independencia: as de um mesmo documento
    compartilham o texto, o autor e o estilo de reporte.
    """
    rng = random.Random(semente)
    docs = list(contagens)
    if not docs:
        return (0.0, 0.0)
    amostras = []
    for _ in range(n):
        sorteados = [rng.choice(docs) for _ in docs]
        tp = sum(contagens[d][f"{prefixo}_tp"] for d in sorteados)
        fp = sum(contagens[d][f"{prefixo}_fp"] for d in sorteados)
        fn = sum(contagens[d][f"{prefixo}_fn"] for d in sorteados)
        amostras.append(prf(tp, fp, fn)["f1"])
    amostras.sort()
    alfa = (1 - nivel) / 2
    return amostras[math.floor(alfa * n)], amostras[math.ceil((1 - alfa) * n) - 1]


def teste_permutacao(
    contagens_a: dict[str, dict],
    contagens_b: dict[str, dict],
    prefixo: str = "tupla",
    n: int = 1000,
    semente: int = 42,
) -> dict:
    """Teste pareado por documento entre dois sistemas.

    A cada permutacao, o resultado de cada documento e trocado entre os dois
    sistemas com probabilidade 1/2. O p-valor e a fracao de permutacoes cuja
    diferenca e ao menos tao extrema quanto a observada.
    """
    docs = sorted(set(contagens_a) & set(contagens_b))

    def f1(contagens, chaves):
        tp = sum(contagens[d][f"{prefixo}_tp"] for d in chaves)
        fp = sum(contagens[d][f"{prefixo}_fp"] for d in chaves)
        fn = sum(contagens[d][f"{prefixo}_fn"] for d in chaves)
        return prf(tp, fp, fn)["f1"]

    observada = f1(contagens_a, docs) - f1(contagens_b, docs)
    rng = random.Random(semente)
    extremas = 0
    for _ in range(n):
        troca = {d: rng.random() < 0.5 for d in docs}
        a = {d: (contagens_b if troca[d] else contagens_a)[d] for d in docs}
        b = {d: (contagens_a if troca[d] else contagens_b)[d] for d in docs}
        if abs(f1(a, docs) - f1(b, docs)) >= abs(observada):
            extremas += 1
    return {"diferenca": observada, "p_valor": (extremas + 1) / (n + 1)}


# ---------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------


def avaliar(
    ouro: dict[str, Registro],
    previsto: dict[str, Registro],
    resumos: dict[str, str] | None = None,
) -> dict:
    contagens = contagens_por_documento(ouro, previsto)
    saida = {
        "documentos": len(contagens),
        "extracoes_ouro": sum(r.n for r in ouro.values()),
        "extracoes_previstas": sum(previsto.get(d, Registro(doc_id=d)).n for d in ouro),
        "deteccao": agregar(contagens, "deteccao"),
        "tupla_completa": agregar(contagens, "tupla"),
        "ic_tupla": bootstrap_ic(contagens, "tupla"),
        "ic_deteccao": bootstrap_ic(contagens, "deteccao"),
        "por_campo": acuracia_por_campo(contagens),
        "contagens": contagens,
    }
    for campo in CAMPOS_CATEGORICOS:
        saida[f"f1_{campo}"] = f1_macro(contagens, campo)
    if resumos is not None:
        saida["alucinacao"] = taxa_alucinacao(previsto, resumos)
    return saida


def imprimir(resultado: dict, titulo: str = "Avaliacao") -> None:
    print(f"\n=== {titulo} ===")
    print(f"documentos           {resultado['documentos']:>6}")
    print(f"extracoes no ouro    {resultado['extracoes_ouro']:>6}")
    print(f"extracoes previstas  {resultado['extracoes_previstas']:>6}")

    for nome, chave, ic in (
        ("Deteccao (o valor foi encontrado)", "deteccao", "ic_deteccao"),
        ("Tupla completa (todos os campos)", "tupla_completa", "ic_tupla"),
    ):
        m = resultado[chave]
        lo, hi = resultado[ic]
        print(f"\n{nome}")
        print(f"  precisao {m['precisao']:.3f}   recall {m['recall']:.3f}   "
              f"F1 {m['f1']:.3f}  IC95% [{lo:.3f}, {hi:.3f}]")
        print(f"  TP {m['tp']}   FP {m['fp']}   FN {m['fn']}")

    print("\nAcerto por campo (entre os pares casados)")
    for campo, v in resultado["por_campo"].items():
        print(f"  {campo:<18} {v['acuracia']:.3f}  ({v['acertos']}/{v['total']})")

    for campo in CAMPOS_CATEGORICOS:
        info = resultado[f"f1_{campo}"]
        print(f"\nF1 macro de {campo}: {info['macro_f1']:.3f}")
        for rotulo, v in sorted(info["por_rotulo"].items(), key=lambda x: str(x[0])):
            print(f"  {rotulo!s:<34} F1 {v['f1']:.3f}  "
                  f"(tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")

    if "alucinacao" in resultado:
        a = resultado["alucinacao"]
        print(f"\nAncoragem: {a['ancoradas']}/{a['total']} literais, "
              f"taxa de alucinacao {a['taxa']:.3f}")