"""Metrica do GEPA: nota por documento e critica em texto.

A nota usa o mesmo alinhamento da avaliacao (`src.avaliar`), para que o GEPA
otimize a mesma coisa que o relatorio mede:

    nota = (F1 de deteccao + F1 de tupla completa) / 2, por documento

Documento sem nada no gold e sem nada na saida vale 1.

A critica diz ao modelo refletor o que faltou, o que sobrou e qual campo saiu
errado, e cita a regra do guia de anotacao que se aplica. As regras sao
genericas (valem para qualquer resumo); os valores citados sao do proprio
documento que o refletor esta lendo. O braco B tem acesso as mesmas duas
fontes: o guia e os erros no dev.

A critica sai em ingles porque as instrucoes que o GEPA reescreve estao em
ingles.
"""

from __future__ import annotations

import dspy

from src.avaliar import CAMPOS_TUPLA, alinhar, campos_iguais, prf, tupla_completa
from src.schema import UNIDADES, Extracao

REGRAS = {
    "listar": (
        "Every number followed by a unit from the vocabulary (%, wt%, ppm, ppb, g/t, "
        "mg/kg, kg/t, ug/g, permil, mg/L, g/L, mol/L) gets an entry, even when it is "
        "not about rare earths; irrelevant ones are listed with metric_type "
        "'Invalid Candidate' instead of being skipped."),
    "fronteira": (
        "Numbers without a vocabulary unit (temperature, pH, years, sample counts, "
        "pressure, mM, mol %, mg/g) get no entry at all."),
    "uma_vez": (
        "Each value is listed once, in its most informative sentence; the "
        "uncertainty after a plus-minus sign is not an entry."),
    "Invalid Candidate": (
        "Invalid Candidate: content of anything that is not a rare earth (major "
        "oxides such as MgO or SiO2, trace elements such as Cr, Ni, Sr, Zr, sulfate, "
        "REE-free minerals), reagent concentrations, additive doses, solid/liquid "
        "ratios, melt fractions, economic values (cut-off grade, price, market share). "
        "ThO2 and U3O8 are the exception: Individual Entity Concentration."),
    "Bulk Concentration": (
        "Bulk Concentration: aggregate rare-earth content (REE, TREE, TREO, REO, LREE, "
        "HREE, REY, sum of REE), including the fraction of the total REE held in a "
        "phase, coming from a source or mobilized by a natural process."),
    "Individual Entity Concentration": (
        "Individual Entity Concentration: content of one specific rare-earth element, "
        "rare-earth oxide or rare-earth mineral, or ThO2/U3O8."),
    "Process Metric": (
        "Process Metric: recovery, leaching or extraction efficiency, selectivity, "
        "separation factor, analytical precision, and rare-earth concentration in "
        "solution."),
    "unit": (
        "unit is the unit as written: % and wt% are different units; the per-mil "
        "sign is permil; microgram per gram is ug/g."),
    "context_modifier": (
        "context_modifier comes only from the words next to this value: about, "
        "approximately, ~ -> approximate; mean, average -> average; up to -> up_to; "
        ">, more than, above -> greater_than; <, less than, below -> less_than; "
        "otherwise none. A modifier before a list applies to the adjacent value only; "
        "plus-minus is not a modifier."),
    "target_entity": (
        "target_entity is the entity as written in the text. For a Process Metric it is "
        "the rare earth named (even if named in the previous sentence), otherwise the "
        "process noun; for an Invalid Candidate it is the noun the number refers to."),
    "entity_type": (
        "entity_type: element for a single element, element_group for REE/TREE/LREE/"
        "HREE and similar, oxide for oxides and TREO/REO, mineral for minerals, other "
        "for anything else."),
    "faixa": (
        "A range such as 'A-B' or 'from A to B' is one entry with value=A, "
        "value_max=B, is_range=true; a single value has value_max=null and "
        "is_range=false."),
    "ancora": (
        "sentence must be copied literally from the abstract, and the value must "
        "appear in that sentence."),
    "formato": 'Answer with the JSON object {"extracoes": [...]} and nothing else.',
}
REGRA_DO_CAMPO = {"value_max": "faixa", "is_range": "faixa", "unit": "unit",
                  "context_modifier": "context_modifier", "target_entity": "target_entity",
                  "entity_type": "entity_type"}


def _num(x: float | None) -> str:
    return "?" if x is None else f"{x:g}"


def descrever(e: Extracao) -> str:
    faixa = _num(e.value) + (f"-{_num(e.value_max)}" if e.value_max is not None else "")
    return (f"{faixa} {e.unit or '(no unit)'} | {e.target_entity} | {e.metric_type} | "
            f"{e.context_modifier} | \"{e.sentence[:90]}\"")


def nota(ouro: list[Extracao], previsto: list[Extracao]) -> tuple[float, float, float]:
    """(nota, F1 de deteccao, F1 de tupla) de um documento."""
    if not ouro and not previsto:
        return 1.0, 1.0, 1.0
    a = alinhar(ouro, previsto)
    completos = sum(tupla_completa(o, p) for o, p in a.pares)
    parciais = len(a.pares) - completos
    det = prf(len(a.pares), len(a.inventadas), len(a.nao_encontradas))["f1"]
    tup = prf(completos, len(a.inventadas) + parciais, len(a.nao_encontradas) + parciais)["f1"]
    return (det + tup) / 2, det, tup


def critica(ouro: list[Extracao], resultado) -> tuple[float, str]:
    """Nota e texto de critica de um documento. `resultado` e um `Resultado`."""
    previsto = resultado.registro.extracoes
    total, det, tup = nota(ouro, previsto)
    linhas = [f"Score {total:.2f} (value detection F1 {det:.2f}, full-entry F1 {tup:.2f}). "
              f"The reference has {len(ouro)} entries; the output kept {len(previsto)}."]
    regras: list[str] = []

    if resultado.erro:
        linhas.append(f"The answer could not be used: {resultado.erro}")
        regras.append("formato")

    a = alinhar(ouro, previsto)
    if a.nao_encontradas:
        linhas.append("Missing entries (in the reference, not in the output):")
        linhas += [f"- {descrever(o)}" for o in a.nao_encontradas]
        regras.append("listar")
        regras += [o.metric_type for o in a.nao_encontradas if o.metric_type == "Invalid Candidate"]
    if a.inventadas:
        linhas.append("Extra entries (in the output, not in the reference):")
        linhas += [f"- {descrever(p)}" for p in a.inventadas]
        fora = any(p.unit not in UNIDADES for p in a.inventadas)
        regras += ["fronteira"] if fora else []
        regras.append("uma_vez")

    erros_campo = []
    for o, p in a.pares:
        for campo in CAMPOS_TUPLA:
            if campo == "value" or campos_iguais(campo, o, p):
                continue
            erros_campo.append(f"- value {_num(o.value)}: {campo} should be {getattr(o, campo)!r}, "
                               f"got {getattr(p, campo)!r}")
            regras.append(o.metric_type if campo == "metric_type" else REGRA_DO_CAMPO[campo])
    if erros_campo:
        linhas.append("Wrong fields on values that were found:")
        linhas += erros_campo

    ancora = [r for r in resultado.rejeitadas if r.etapa == "ancoragem"]
    if ancora:
        linhas.append(f"{len(ancora)} entries were discarded because the sentence or the value "
                      "is not in the abstract.")
        regras.append("ancora")
    schema = [r for r in resultado.rejeitadas if r.etapa == "schema"]
    if schema:
        linhas.append(f"{len(schema)} entries were discarded by the schema: "
                      + "; ".join(r.motivo for r in schema[:3]))
        if any("unidade" in r.motivo for r in schema):
            regras.append("fronteira")

    if total == 1.0 and not erros_campo:
        linhas.append("All entries match the reference.")
    unicas = list(dict.fromkeys(regras))
    if unicas:
        linhas.append("Annotation rules that apply:")
        linhas += [f"* {REGRAS[r]}" for r in unicas]
    return total, "\n".join(linhas)


def metrica_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Assinatura exigida pelo dspy.GEPA. `gold.ouro` e a lista de extracoes do gold."""
    ouro = [Extracao(**e) for e in gold.ouro]
    total, texto = critica(ouro, pred.resultado)
    return dspy.Prediction(score=total, feedback=texto)


def metrica_simples(gold, pred, trace=None) -> float:
    return nota([Extracao(**e) for e in gold.ouro], pred.resultado.registro.extracoes)[0]
