#!/usr/bin/env python3
"""Analise final: tabelas, testes de permutacao e as figuras de resultado.

    python scripts/analisar_protegi.py --gold ../gold_standard/gold_final.jsonl

ADAPTADO do analisar.py original (braco C = GEPA) para o braco C = ProTeGi
(ver conversa: "PROTEGI SUBSTITUI O GEPA... serao dois conjuntos de bracos
A,B,C", um por algoritmo). Duas diferencas do original:

  1. Sem particionar.py (nao existe neste projeto): o gold e carregado e
     particionado com as funcoes de apo.py (carregar_gold +
     dividir_gold_por_particao), que ja fazem exatamente isso a partir do
     mesmo gold_final.jsonl -- nao precisa reimplementar nem duplicar.
  2. BRACOS aponta "c_protegi" em vez de "c_gepa", e a trajetoria vem de
     results/protegi/c_protegi/candidatos.json em vez de results/gepa/...
     Saidas ganham sufixo "_protegi" (fig1_protegi.pdf, etc.) para nunca
     colidir com as saidas do analisar.py do colaborador (braco C = GEPA),
     caso os dois rodem sobre a mesma pasta results/ compartilhada.

Le (o que existir):
    results/test/<prompt>_s<semente>.jsonl      bracos no teste
    results/val/<prompt>_s<semente>.jsonl        A, B e C no val (fig 1 e 5)
    results/protegi/c_protegi/candidatos.json   trajetoria do ProTeGi
    data/gold/piloto_r1.jsonl, piloto_r2.jsonl  teto do anotador (opcional)

Grava em results/analise/:
    analise_protegi.json        todos os numeros
    tabela_teste_protegi.tex    linhas da tabela principal do relatorio
    fig1_protegi.pdf            trajetoria do ProTeGi no val
    fig2_f1.pdf                 F1 de deteccao e de tupla no teste
    fig3_kappa.pdf               kappa por campo: modelo x gold e anotador x anotador
    fig4_confusao.pdf           matriz de confusao de metric_type no teste
    fig5_val_teste.pdf          nota no val x nota no teste
(e um .png de cada figura, para conferir)

Unidades. A "nota" e a do GEPA/ProTeGi: media por documento de (F1 deteccao +
F1 tupla) / 2 (src.critica.nota -- mesma formula para os dois algoritmos, so
muda quem gerou o prompt). O kappa do modelo e calculado exatamente como o do
anotador (scripts/concordancia.py), com o gold no papel da rodada 1 e o
modelo no da rodada 2. Assim cada comparacao com o teto e feita na mesma
unidade.

Agregacao das sementes: as contagens de cada documento sao somadas nas tres
sementes, e o IC vem de bootstrap por documento sobre essas somas.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from shutil import which
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from concordancia import carregar as carregar_rodada_anotador, concordancia, kappa  # noqa: E402
from src.avaliar import (agregar, alinhar, bootstrap_ic, contagens_por_documento,  # noqa: E402
                         teste_permutacao)
from src.critica import nota  # noqa: E402
from src.extrair import carregar_resultados_llm  # noqa: E402
from src.schema import Registro  # noqa: E402

import apo  # scripts/apo.py -- so para carregar/particionar o gold, ver docstring  # noqa: E402

RESULTADOS = RAIZ / "results"
SAIDA = RESULTADOS / "analise"
GOLD_DIR = RAIZ / "data" / "gold"
SEMENTES = (0, 1, 2)

# (chave, arquivo do prompt, rotulo nas figuras) -- C = ProTeGi neste script.
BRACOS = [
    ("A", "v00_ingenuo", "A"),
    ("B", "b_regras", "B"),
    ("C", "c_protegi", "C"),
]
CATEGORICOS = ("metric_type", "entity_type", "context_modifier", "unit")
METRICAS = ["Bulk Concentration", "Individual Entity Concentration", "Process Metric",
            "Invalid Candidate"]
ROTULO_METRICA = {"Bulk Concentration": "Bulk", "Individual Entity Concentration": "Individual",
                  "Process Metric": "Process", "Invalid Candidate": "Invalid"}

# Paleta categorica de referencia (slots 1-3, validados em todos os pares) e tintas.
COR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
MARCADOR = {"A": "o", "B": "s", "C": "^"}
SUPERFICIE = "white"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8983"
GRADE = "#e6e5e0"
SEQUENCIAL = LinearSegmentedColormap.from_list("azul", ["#ffffff", "#b9d3f2", "#2a78d6", "#0d3a73"])

# Estilo de plotagem do projeto
USETEX = all(which(p) for p in ("latex", "dvipng"))
rcParams.update({
    "text.usetex": USETEX,
    "font.family": "serif",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "figure.dpi": 135,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})
RETESTE = "Teste-reteste"
COLUNA = 5.0  # figura de uma coluna; no LaTeX vai a \\columnwidth
PAGINA = 10.5

ARGS = None  # preenchido em main(); ver carregar_gold()


# ---------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------


def carregar_gold(particao: str) -> dict[str, Registro]:
    """particao: "dev" (treino), "val" ou "test" -- via apo.py, nao
    particionar.py (que nao existe neste projeto). O split vem sempre do
    campo "particao" do proprio gold_final.jsonl, igual ao apo.py usa em
    `otimizar`/`avaliar-prompt`."""
    gold_bruto = apo.carregar_gold(Path(ARGS.gold))
    treino, val, teste = apo.dividir_gold_por_particao(gold_bruto)
    mapa = {"dev": treino, "val": val, "test": teste}
    return {d: Registro(doc_id=d, extracoes=ex) for d, ex in mapa[particao].items()}


def carregar_previsto(caminho: Path, gold: dict[str, Registro]) -> dict[str, Registro]:
    previsto = carregar_resultados_llm(caminho)
    return {d: previsto.get(d, Registro(doc_id=d)) for d in gold}


def nota_media(gold: dict[str, Registro], previsto: dict[str, Registro]) -> float:
    return statistics.mean(nota(gold[d].extracoes, previsto[d].extracoes)[0] for d in gold)


def somar_sementes(contagens: list[dict[str, dict]]) -> dict[str, dict]:
    """Contagens por documento somadas nas sementes; os pares sao concatenados."""
    soma = {}
    for d in contagens[0]:
        soma[d] = {k: sum(c[d][k] for c in contagens) for k in contagens[0][d] if k != "pares"}
        soma[d]["pares"] = [p for c in contagens for p in c[d]["pares"]]
    return soma


def kappas(pares: list) -> dict[str, float | None]:
    return {c: kappa([(getattr(o, c), getattr(p, c)) for o, p in pares]) for c in CATEGORICOS}


def confusao(gold: dict[str, Registro], rodadas: list[dict[str, Registro]]) -> Counter:
    """(classe do gold, classe prevista); 'ausente' marca o que so existe de um lado."""
    matriz = Counter()
    for previsto in rodadas:
        for d in gold:
            a = alinhar(gold[d].extracoes, previsto[d].extracoes)
            matriz.update((o.metric_type, p.metric_type) for o, p in a.pares)
            matriz.update((o.metric_type, "ausente") for o in a.nao_encontradas)
            matriz.update(("ausente", p.metric_type) for p in a.inventadas)
    return matriz


def avaliar_braco(gold: dict[str, Registro], prompt: str) -> dict | None:
    arquivos = [RESULTADOS / "test" / f"{prompt}_s{s}.jsonl" for s in SEMENTES]
    arquivos = [a for a in arquivos if a.exists()]
    if not arquivos:
        return None
    rodadas = [carregar_previsto(a, gold) for a in arquivos]
    contagens = [contagens_por_documento(gold, r) for r in rodadas]
    por_semente = []
    for r, c in zip(rodadas, contagens):
        pares = [p for x in c.values() for p in x["pares"]]
        por_semente.append({"deteccao": agregar(c, "deteccao")["f1"],
                            "tupla": agregar(c, "tupla")["f1"],
                            "nota": nota_media(gold, r), "kappa": kappas(pares)})
    soma = somar_sementes(contagens)
    return {
        "sementes": len(rodadas),
        "por_semente": por_semente,
        "deteccao": agregar(soma, "deteccao"),
        "ic_deteccao": bootstrap_ic(soma, "deteccao"),
        "tupla": agregar(soma, "tupla"),
        "ic_tupla": bootstrap_ic(soma, "tupla"),
        "nota": statistics.mean(s["nota"] for s in por_semente),
        "kappa": kappas([p for x in soma.values() for p in x["pares"]]),
        "confusao": {f"{a}|{b}": n for (a, b), n in confusao(gold, rodadas).items()},
        "_contagens": soma,
    }


def avaliar_anotador() -> dict | None:
    r1, r2 = GOLD_DIR / "piloto_r1.jsonl", GOLD_DIR / "piloto_r2.jsonl"
    if not (r1.exists() and r2.exists()):
        return None
    a, b = carregar_rodada_anotador(r1), carregar_rodada_anotador(r2)
    res = concordancia(a, b)
    comuns = sorted(set(a) & set(b))
    return {
        "documentos": res["documentos"],
        "deteccao": res["deteccao"], "ic_deteccao": res["ic_deteccao"],
        "tupla": res["tupla"], "ic_tupla": res["ic_tupla"],
        "kappa": {c: res["campos"][c]["kappa"] for c in CATEGORICOS},
        "nota": statistics.mean(nota(a[d].extracoes, b[d].extracoes)[0] for d in comuns),
    }


def nota_val_braco(prompt: str) -> float | None:
    """Nota no val (mesma formula e mesmo pipeline do teste): media das
    sementes em results/val/<prompt>_s<seed>.jsonl, geradas por
    `rodar_bracos.py --tambem-validacao <prompt>`. None se nao existir nenhuma."""
    arquivos = [RESULTADOS / "val" / f"{prompt}_s{s}.jsonl" for s in SEMENTES]
    arquivos = [a for a in arquivos if a.exists()]
    if not arquivos:
        return None
    gold = carregar_gold("val")
    return statistics.mean(nota_media(gold, carregar_previsto(a, gold)) for a in arquivos)


def nota_val_b_regras() -> float | None:
    return nota_val_braco("b_regras")


# ---------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------


def salvar(fig, nome: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(SAIDA / f"{nome}.{ext}")
    plt.close(fig)


def fmt(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def fig_protegi(protegi: dict, anotador: dict | None, val_b: float | None) -> None:
    """Trajetoria da busca do ProTeGi no val -- le exatamente o
    candidatos.json que `apo.py otimizar` gera (indice/pais/passo/no_beam/
    nota_val/nota_treino/chamadas_ate_aqui/instrucoes)."""
    cand = protegi["candidatas"]
    x = {c["indice"]: c["chamadas_ate_aqui"] for c in cand}
    y = {c["indice"]: c["nota_val"] for c in cand}
    fig, ax = plt.subplots(figsize=(COLUNA, 3.6))
    for c in cand:  # arestas pai -> filho
        for pai in c["pais"] or []:
            if pai is not None:
                ax.plot([x[pai], x[c["indice"]]], [y[pai], y[c["indice"]]],
                        color=GRADE, linewidth=1.2, zorder=1)
    ordem = sorted(cand, key=lambda c: c["chamadas_ate_aqui"])
    melhor, xs, ys = -1.0, [], []
    for c in ordem:
        melhor = max(melhor, c["nota_val"])
        xs.append(c["chamadas_ate_aqui"])
        ys.append(melhor)
    xs.append(protegi["chamadas_metrica"])
    ys.append(ys[-1])
    ax.step(xs, ys, where="post", color=COR["C"], linewidth=2, zorder=2,
            label="C: Melhor candidato")
    ax.scatter(list(x.values()), list(y.values()), s=28, color=COR["C"],
               edgecolor=SUPERFICIE, linewidth=1.2, zorder=3, label="C: Candidatos")
    referencias = [(y[0], COR["A"], (0, (6, 3)), "A")]
    if val_b is not None:
        referencias.append((val_b, COR["B"], "-.", "B"))
    if anotador:
        referencias.append((anotador["nota"], TINTA_2, (0, (1.5, 2.5)), RETESTE))
    for valor, cor, estilo, rotulo in referencias:
        ax.axhline(valor, color=cor, linestyle=estilo, linewidth=1.4, zorder=1,
                   label=f"{rotulo} = {fmt(valor)}")
    valores = [*y.values(), *(r[0] for r in referencias)]
    ax.set_ylim(min(valores) - 0.03, max(valores) + 0.05)
    ax.set_xlim(-8, protegi["chamadas_metrica"] + 4)
    ax.set_xlabel("Documentos avaliados")
    ax.set_ylabel("F1-score m\u00e9dio na valida\u00e7\u00e3o")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6,
              fontsize="small", columnspacing=1.0)
    salvar(fig, "fig1_protegi")


def fig_f1(resultados: dict, anotador: dict | None) -> None:
    bracos = [b for b in BRACOS if b[0] in resultados]
    fig, eixos = plt.subplots(1, 2, figsize=(COLUNA, 3.4), sharey=True)
    for ax, (chave, titulo) in zip(eixos, (("deteccao", "Detec\u00e7\u00e3o"),
                                           ("tupla", "Tupla completa"))):
        for i, (b, _, rotulo) in enumerate(bracos):
            r = resultados[b]
            f1 = r[chave]["f1"]
            lo, hi = r[f"ic_{chave}"]
            ax.bar(i, f1, width=0.62, color=COR[b], edgecolor=SUPERFICIE, linewidth=1)
            ax.errorbar(i, f1, yerr=[[f1 - lo], [hi - f1]], color=TINTA, linewidth=0.9,
                        capsize=2.5)
            sementes = [s[chave] for s in r["por_semente"]]
            ax.scatter([i + d for d in (-0.12, 0, 0.12)][:len(sementes)], sementes, s=9,
                       color=SUPERFICIE, edgecolor=TINTA, linewidth=0.7, zorder=4)
            ax.text(i, hi + 0.02, fmt(f1), ha="center", va="bottom", fontsize="small")
        if anotador and chave == "deteccao":
            v = anotador["deteccao"]["f1"]
            ax.axhline(v, color=TINTA_2, linestyle=":", linewidth=1.2,
                       label=f"{RETESTE} = {fmt(v)}")
            fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), handlelength=1.6)
        ax.set_xticks(range(len(bracos)), [b[0] for b in bracos])
        ax.set_title(titulo)
        ax.set_ylim(0, 1.1)
    eixos[0].set_ylabel("F1-score no teste")
    salvar(fig, "fig2_f1")


def fig_kappa(resultados: dict, anotador: dict | None) -> None:
    bracos = [b for b in BRACOS if b[0] in resultados]
    n = len(bracos)
    largura = 0.8 / n
    fig, ax = plt.subplots(figsize=(COLUNA, 3.6))
    for j, (b, _, rotulo) in enumerate(bracos):
        for i, campo in enumerate(CATEGORICOS):
            k = resultados[b]["kappa"][campo] or 0.0
            xi = i - 0.4 + largura * (j + 0.5)
            ax.bar(xi, k, width=largura, color=COR[b], edgecolor=SUPERFICIE, linewidth=1,
                   label=rotulo if i == 0 else None)
            ks = [s["kappa"][campo] for s in resultados[b]["por_semente"]
                  if s["kappa"][campo] is not None]
            if len(ks) > 1:
                ax.plot([xi, xi], [min(ks), max(ks)], color=TINTA, linewidth=0.9)
            ax.text(xi, max([k, *ks]) + 0.015, fmt(k), ha="center", va="bottom",
                    fontsize="x-small", rotation=90)
    if anotador:
        for i, campo in enumerate(CATEGORICOS):
            ax.plot([i - 0.44, i + 0.44], [anotador["kappa"][campo]] * 2, color=TINTA,
                    linewidth=1.6, linestyle=":", label=RETESTE if i == 0 else None)
    ax.set_xticks(range(len(CATEGORICOS)), [c.replace("_", "\n", 1) for c in CATEGORICOS])
    ax.set_ylabel("$\\kappa$ de Cohen")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.0,
              handlelength=1.4)
    salvar(fig, "fig3_kappa")


def fig_confusao(resultados: dict) -> None:
    bracos = [b for b in BRACOS if b[0] in resultados]
    linhas = METRICAS + ["ausente"]
    colunas = METRICAS + ["ausente"]
    rotulo_linha = [ROTULO_METRICA[m] for m in METRICAS] + ["(fora do gold)"]
    rotulo_coluna = [ROTULO_METRICA[m] for m in METRICAS] + ["(n\u00e3o achou)"]
    fig, eixos = plt.subplots(1, len(bracos), figsize=(PAGINA, 3.6), squeeze=False)
    for ax, (b, _, rotulo) in zip(eixos[0], bracos):
        conf = {tuple(k.split("|")): v for k, v in resultados[b]["confusao"].items()}
        for i, lin in enumerate(linhas):
            total = sum(conf.get((lin, c), 0) for c in colunas) or 1
            for j, col in enumerate(colunas):
                n = conf.get((lin, col), 0)
                frac = n / total
                if lin == "ausente" and col == "ausente":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRADE))
                    continue
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=SEQUENCIAL(frac),
                                           ec=SUPERFICIE, lw=1.5))
                if n:
                    ax.text(j, i, str(n), ha="center", va="center", fontsize="small",
                            color=SUPERFICIE if frac > 0.55 else TINTA)
        ax.set_xlim(-0.5, len(colunas) - 0.5)
        ax.set_ylim(len(linhas) - 0.5, -0.5)
        ax.set_xticks(range(len(colunas)), rotulo_coluna, rotation=35, ha="right")
        ax.set_yticks(range(len(linhas)), rotulo_linha if ax is eixos[0][0] else [])
        ax.set_title(rotulo)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("Previsto")
    eixos[0][0].set_ylabel("Gold")
    salvar(fig, "fig4_confusao")


def fig_val_teste(val: dict[str, float], resultados: dict) -> None:
    bracos = [b for b in BRACOS if b[0] in resultados and b[0] in val]
    fig, ax = plt.subplots(figsize=(COLUNA * 0.75, 3.6))
    for b, _, rotulo in bracos:
        v, t = val[b], resultados[b]["nota"]
        ax.plot([0, 1], [v, t], color=COR[b], linewidth=2, marker=MARCADOR[b], markersize=7,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.2,
                label=f"{rotulo}: {fmt(v)} $\\to$ {fmt(t)}")
    ax.set_xticks([0, 1], ["Valida\u00e7\u00e3o", "Teste"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("F1-score m\u00e9dio")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), handlelength=1.6)
    salvar(fig, "fig5_val_teste")


# ---------------------------------------------------------------------
# Tabela e main
# ---------------------------------------------------------------------


def linha_tex(nome: str, r: dict) -> str:
    def f1(chave):
        lo, hi = r[f"ic_{chave}"]
        return f"{r[chave]['f1']:.2f} [{lo:.2f}; {hi:.2f}]".replace(".", ",")
    ks = " & ".join("--" if r["kappa"][c] is None else f"{r['kappa'][c]:.2f}".replace(".", ",")
                    for c in CATEGORICOS)
    return f"{nome} & {f1('deteccao')} & {f1('tupla')} & {r['nota']:.2f}".replace(".", ",") \
        + f" & {ks} \\\\"


def main() -> int:
    global ARGS, RESULTADOS, SAIDA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True,
                     help="gold_final.jsonl (doc_id, extracoes, particao) -- mesmo arquivo usado "
                          "em apo.py otimizar/avaliar-prompt e em rodar_bracos.py.")
    ap.add_argument("--results-dir", default=str(RESULTADOS), dest="results_dir",
                     help="Pasta 'results/' onde estao test/, val/ e protegi/c_protegi/candidatos.json.")
    ARGS = ap.parse_args()

    RESULTADOS = Path(ARGS.results_dir)
    SAIDA = RESULTADOS / "analise"
    SAIDA.mkdir(parents=True, exist_ok=True)

    gold = carregar_gold("test")
    resultados = {}
    for b, prompt, _ in BRACOS:
        r = avaliar_braco(gold, prompt)
        if r:
            resultados[b] = r
            print(f"{b:<3} {r['sementes']} sementes  deteccao {r['deteccao']['f1']:.3f}  "
                  f"tupla {r['tupla']['f1']:.3f}  nota {r['nota']:.3f}")
    anotador = avaliar_anotador()
    caminho_protegi = RESULTADOS / "protegi" / "c_protegi" / "candidatos.json"
    protegi = json.loads(caminho_protegi.read_text(encoding="utf-8")) if caminho_protegi.exists() else None
    val_b = nota_val_b_regras()

    permutacoes = {}
    chaves = [b for b, _, _ in BRACOS if b in resultados]
    for i, x in enumerate(chaves):
        for y in chaves[i + 1:]:
            permutacoes[f"{y} - {x}"] = {
                m: teste_permutacao(resultados[y]["_contagens"], resultados[x]["_contagens"], m)
                for m in ("deteccao", "tupla")}

    if protegi:
        fig_protegi(protegi, anotador, val_b)
    if resultados:
        fig_f1(resultados, anotador)
        fig_kappa(resultados, anotador)
        fig_confusao(resultados)
    if protegi and resultados:
        # preferencia: nota do val recalculada com o MESMO pipeline/formula do
        # teste (results/val/*.jsonl). Fallback: nota_val do candidatos.json,
        # que e so F1 de deteccao com a extracao simples do apo.py -- nao e
        # diretamente comparavel com a nota do teste.
        val = {}
        fallback = {"A": protegi["candidatas"][0]["nota_val"],
                    "C": protegi["candidatas"][protegi["melhor"]]["nota_val"]}
        for b, prompt, _ in BRACOS:
            v = nota_val_braco(prompt)
            if v is not None:
                val[b] = v
            elif b in fallback:
                val[b] = fallback[b]
                print(f"[aviso] {b}: sem results/val/{prompt}_s*.jsonl; fig5 usa nota_val do "
                      f"candidatos.json (F1 de deteccao, extracao simples) -- nao comparavel com o teste.")
        fig_val_teste(val, resultados)

    linhas = [linha_tex(b, resultados[b]) for b in chaves]
    if anotador:
        linhas.append(linha_tex("reteste*", anotador))
    (SAIDA / "tabela_teste_protegi.tex").write_text("\n".join(linhas) + "\n", encoding="ascii")

    publico = {b: {k: v for k, v in r.items() if not k.startswith("_")}
               for b, r in resultados.items()}
    (SAIDA / "analise_protegi.json").write_text(json.dumps(
        {"teste": publico, "anotador": anotador, "permutacao": permutacoes,
         "val_b_regras": val_b}, indent=1, ensure_ascii=True), encoding="ascii")

    for par, m in permutacoes.items():
        print(f"{par:<8} " + "  ".join(f"{k} dif {v['diferenca']:+.3f} p {v['p_valor']:.3f}"
                                       for k, v in m.items()))
    print(f"\nsaidas em {SAIDA}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())