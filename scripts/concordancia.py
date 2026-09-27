#!/usr/bin/env python3
"""Concordancia teste-reteste do anotador (piloto_r1 x piloto_r2).

    python scripts/concordancia.py
    python scripts/concordancia.py --saida results/concordancia.txt

As duas rodadas sao alinhadas pelo mesmo casamento guloso da avaliacao
(`src.avaliar.alinhar`), com a r1 no papel de referencia. Saem tres leituras:

  deteccao     os dois anotaram o mesmo valor? (F1, simetrico)
  tupla        anotaram o mesmo valor com todos os campos iguais?
  por campo    entre os pares casados, acordo bruto e kappa de Cohen dos
               campos categoricos (metric_type, entity_type, context_modifier,
               unit)

O kappa desconta o acordo esperado ao acaso, dado quanto cada rodada usa cada
rotulo. E o teto de referencia para os bracos: nenhum prompt pode ser cobrado
por concordar com o gold mais do que o proprio anotador concorda consigo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.avaliar import CAMPOS_TUPLA, agregar, alinhar, bootstrap_ic, campos_iguais, \
    contagens_por_documento  # noqa: E402
from src.schema import Registro  # noqa: E402

GOLD = RAIZ / "data" / "gold"
CATEGORICOS = ("metric_type", "entity_type", "context_modifier", "unit")


def carregar(caminho: Path) -> dict[str, Registro]:
    regs = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            r = json.loads(linha)
            regs[r["doc_id"]] = Registro(doc_id=r["doc_id"], extracoes=r["extracoes"])
    return regs


def kappa(pares: list[tuple]) -> float | None:
    """Kappa de Cohen para uma lista de pares (rotulo_r1, rotulo_r2)."""
    n = len(pares)
    if n == 0:
        return None
    po = sum(a == b for a, b in pares) / n
    c1, c2 = Counter(a for a, _ in pares), Counter(b for _, b in pares)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def concordancia(r1: dict[str, Registro], r2: dict[str, Registro]) -> dict:
    comuns = sorted(set(r1) & set(r2))
    a, b = {d: r1[d] for d in comuns}, {d: r2[d] for d in comuns}
    contagens = contagens_por_documento(a, b)
    pares = [p for c in contagens.values() for p in c["pares"]]
    campos = {}
    for campo in CAMPOS_TUPLA:
        iguais = sum(campos_iguais(campo, o, p) for o, p in pares)
        campos[campo] = {"acordo": iguais / len(pares) if pares else 0.0, "n": len(pares)}
        if campo in CATEGORICOS:
            campos[campo]["kappa"] = kappa([(getattr(o, campo), getattr(p, campo))
                                            for o, p in pares])
    return {
        "documentos": len(comuns),
        "extracoes_r1": sum(r.n for r in a.values()),
        "extracoes_r2": sum(r.n for r in b.values()),
        "deteccao": agregar(contagens, "deteccao"),
        "ic_deteccao": bootstrap_ic(contagens, "deteccao"),
        "tupla": agregar(contagens, "tupla"),
        "ic_tupla": bootstrap_ic(contagens, "tupla"),
        "campos": campos,
    }


def resumo_ext(e) -> str:
    faixa = f"{e.value:g}-{e.value_max:g}" if e.value_max is not None else f"{e.value:g}"
    return (f"{faixa} {e.unit or '-'} | {e.target_entity} ({e.entity_type}) | "
            f"{e.metric_type} | {e.context_modifier}")


def divergencias(r1: dict[str, Registro], r2: dict[str, Registro]) -> list[str]:
    linhas = []
    for d in sorted(set(r1) & set(r2)):
        al = alinhar(r1[d].extracoes, r2[d].extracoes)
        bloco = []
        for o, p in al.pares:
            dif = [c for c in CAMPOS_TUPLA if not campos_iguais(c, o, p)]
            if dif:
                bloco.append(f"  DIVERGE  {resumo_ext(o)}")
                bloco += [f"           {c}: r1 {getattr(o, c)!r} | r2 {getattr(p, c)!r}" for c in dif]
        bloco += [f"  SO NA R1 {resumo_ext(o)}" for o in al.nao_encontradas]
        bloco += [f"  SO NA R2 {resumo_ext(p)}" for p in al.inventadas]
        if bloco:
            linhas += [f"=== {d}", *bloco, ""]
    return linhas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concordancia teste-reteste")
    parser.add_argument("--r1", type=Path, default=GOLD / "piloto_r1.jsonl")
    parser.add_argument("--r2", type=Path, default=GOLD / "piloto_r2.jsonl")
    parser.add_argument("--saida", type=Path, default=RAIZ / "results" / "concordancia.txt")
    args = parser.parse_args(argv)

    r1, r2 = carregar(args.r1), carregar(args.r2)
    res = concordancia(r1, r2)
    linhas = [
        f"documentos em comum  {res['documentos']}",
        f"extracoes            r1 {res['extracoes_r1']}   r2 {res['extracoes_r2']}",
        "",
    ]
    for nome, chave, ic in (("deteccao (mesmo valor)", "deteccao", "ic_deteccao"),
                            ("tupla completa", "tupla", "ic_tupla")):
        m, (lo, hi) = res[chave], res[ic]
        linhas.append(f"{nome:<24} F1 {m['f1']:.3f}  IC95% [{lo:.3f}, {hi:.3f}]  "
                      f"(casados {m['tp']}, so r2 {m['fp']}, so r1 {m['fn']})")
    linhas += ["", f"{'campo':<18} {'acordo':>7} {'kappa':>7}   (entre {res['campos']['value']['n']} pares)"]
    for campo, v in res["campos"].items():
        k = v.get("kappa")
        linhas.append(f"{campo:<18} {v['acordo']:>7.3f} {'' if k is None else f'{k:>7.3f}'}")
    texto = "\n".join(linhas)
    print(texto)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(texto + "\n\n" + "\n".join(divergencias(r1, r2)), encoding="utf-8")
    print(f"\ndivergencias documento a documento: {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
