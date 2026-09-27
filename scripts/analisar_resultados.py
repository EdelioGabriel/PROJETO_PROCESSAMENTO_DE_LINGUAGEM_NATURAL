#!/usr/bin/env python3
"""Analise final: tabelas, teste de permutacao e figuras de resultado.

Compara o braco A (prompt ingenuo, sem otimizacao) com o braco C (prompt
final produzido pelo ProTeGi), no holdout de TESTE.

    python analisar_resultados.py \\
        --gold ../gold_standard/gold_final.jsonl \\
        --output-dir ../outputs/output_apo_oficial_hpc/results \\
        --candidatos-json ../outputs/output_apo_oficial_hpc/candidatos.json \\
        --saida ../outputs/output_apo_oficial_hpc/results/analise

Le (o que existir):
    <output-dir>/test/v00_ingenuo_s<semente>.jsonl   braco A no teste
    <output-dir>/test/c_protegi_s<semente>.jsonl     braco C no teste
    <output-dir>/val/c_protegi_s0.jsonl              C na validacao (opcional, fig5)
    --candidatos-json                                trajetoria do ProTeGi (opcional, fig1/fig5)
    data/gold/piloto_r1.jsonl, piloto_r2.jsonl        teto do anotador (opcional)

Grava em --saida:
    analise.json        todos os numeros
    tabela_teste.tex     linhas da tabela principal do relatorio
    fig1_protegi.pdf     trajetoria do ProTeGi no val (so se --candidatos-json)
    fig2_f1.pdf          F1 de deteccao e de tupla no teste
    fig3_kappa.pdf       kappa por campo: modelo x gold (e anotador x anotador, se existir)
    fig4_confusao.pdf    matriz de confusao de metric_type no teste
    fig5_val_teste.pdf   nota no val x nota no teste (so se --candidatos-json)
(e um .png de cada figura, para conferir)

Unidades. A "nota" e media por documento de (F1 deteccao + F1 tupla) / 2,
via src.critica.nota. O kappa do modelo e calculado exatamente como o do
anotador (scripts/concordancia.py), com o gold no papel da rodada 1 e o
modelo no da rodada 2 -- so entra na analise se piloto_r1/r2 existirem.

Agregacao das sementes: as contagens de cada documento sao somadas nas tres
sementes, e o IC vem de bootstrap por documento sobre essas somas.

Este script e uma adaptacao de analisar.py (experimento anterior, A/B/C =
ingenuo/regras/GEPA). Aqui nao ha braco B: a arquitetura do ProTeGi (busca
por gradiente textual + beam, sem populacao/merge) nao produz um prompt
"de regras" comparavel ao que o experimento GEPA usava como B -- por isso
B foi removido, em vez de forcado a existir so pra preencher a tabela.
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

# RAIZ = pasta acima de scripts/ (mesmo layout do rodar_bracos.py / apo.py).
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from src.avaliar import (agregar, alinhar, bootstrap_ic, contagens_por_documento,  # noqa: E402
                         teste_permutacao)
from src.critica import nota  # noqa: E402
from src.extrair import carregar_resultados_llm  # noqa: E402
from src.schema import Registro  # noqa: E402

# concordancia.py e opcional: so e necessario se voce tiver piloto_r1/r2
# (estudo de concordancia entre anotadores). Sem ele, kappa/nota do
# anotador simplesmente nao aparecem na analise.
try:
    from concordancia import carregar as carregar_rodada_anotador, concordancia, kappa  # noqa: E402
    TEM_CONCORDANCIA = True
except ImportError:
    TEM_CONCORDANCIA = False

SEMENTES = (0, 1, 2)

# (chave, rotulo-nos-arquivos, rotulo-nas-figuras). Braco C e o unico
# parametrizavel (--braco-c); A e sempre o ingenuo fixo do experimento.
ROTULO_ARQUIVO_A = "v00_ingenuo"

CATEGORICOS = ("metric_type", "entity_type", "context_modifier", "unit")
METRICAS = ["Bulk Concentration", "Individual Entity Concentration", "Process Metric",
            "Invalid Candidate"]
ROTULO_METRICA = {"Bulk Concentration": "Bulk", "Individual Entity Concentration": "Individual",
                  "Process Metric": "Process", "Invalid Candidate": "Invalid"}

# Paleta categorica: A e C (sem B nesta rodada).
COR = {"A": "#2a78d6", "C": "#1baf7a"}
MARCADOR = {"A": "o", "C": "^"}
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


# ---------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------


def carregar_gold_particao(caminho_gold: Path, particao_alvo: str) -> dict[str, Registro]:
    """Le --gold (doc_id, extracoes, particao) e filtra pela particao
    (valores observados: "dev", "val", "test" -- ver apo.PARTICAO_PARA_CONJUNTO).
    """
    saida = {}
    with caminho_gold.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            r = json.loads(linha)
            if r.get("particao") == particao_alvo:
                saida[r["doc_id"]] = Registro(doc_id=r["doc_id"], extracoes=r["extracoes"])
    return saida


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


def avaliar_braco(gold: dict[str, Registro], results_dir: Path, rotulo_arquivo: str) -> dict | None:
    arquivos = [results_dir / "test" / f"{rotulo_arquivo}_s{s}.jsonl" for s in SEMENTES]
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
                            "nota": nota_media(gold, r),
                            "kappa": kappas(pares) if TEM_CONCORDANCIA else {}})
    soma = somar_sementes(contagens)
    return {
        "sementes": len(rodadas),
        "por_semente": por_semente,
        "deteccao": agregar(soma, "deteccao"),
        "ic_deteccao": bootstrap_ic(soma, "deteccao"),
        "tupla": agregar(soma, "tupla"),
        "ic_tupla": bootstrap_ic(soma, "tupla"),
        "nota": statistics.mean(s["nota"] for s in por_semente),
        "kappa": kappas([p for x in soma.values() for p in x["pares"]]) if TEM_CONCORDANCIA else {},
        "confusao": {f"{a}|{b}": n for (a, b), n in confusao(gold, rodadas).items()},
        "_contagens": soma,
    }


def avaliar_anotador(gold_dir: Path | None) -> dict | None:
    if gold_dir is None or not TEM_CONCORDANCIA:
        return None
    r1, r2 = gold_dir / "piloto_r1.jsonl", gold_dir / "piloto_r2.jsonl"
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


def nota_val_braco_c(results_dir: Path, rotulo_arquivo_c: str,
                      gold_val: dict[str, Registro]) -> float | None:
    caminho = results_dir / "val" / f"{rotulo_arquivo_c}_s0.jsonl"
    if not caminho.exists() or not gold_val:
        return None
    return nota_media(gold_val, carregar_previsto(caminho, gold_val))


# ---------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------


def salvar(fig, nome: str, saida_dir: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(saida_dir / f"{nome}.{ext}")
    plt.close(fig)


def fmt(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def fig_protegi(trajetoria_json: dict, anotador: dict | None, bracos_labels: list,
                 saida_dir: Path) -> None:
    """Trajetoria da busca ProTeGi no val, com A e C (final) como referencias.

    Reaproveita o formato de candidatos.json compartilhado por ProTeGi e
    GEPA (ver comentario em apo.py:otimizar_protegi): "candidatas", cada
    uma com "indice", "pais", "nota_val", "chamadas_ate_aqui". "pais" no
    ProTeGi tem sempre um unico elemento (edicao de um candidato), mas o
    schema e o mesmo usado pelo analisar.py original.
    """
    cand = trajetoria_json["candidatas"]
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
    xs.append(trajetoria_json["chamadas_metrica"])
    ys.append(ys[-1])
    ax.step(xs, ys, where="post", color=COR["C"], linewidth=2, zorder=2,
            label="C: Melhor candidato")
    ax.scatter(list(x.values()), list(y.values()), s=28, color=COR["C"],
               edgecolor=SUPERFICIE, linewidth=1.2, zorder=3, label="C: Candidatos")
    referencias = [(y[0], COR["A"], (0, (6, 3)), "A")] if 0 in y else []
    if anotador:
        referencias.append((anotador["nota"], TINTA_2, (0, (1.5, 2.5)), RETESTE))
    for valor, cor, estilo, rotulo in referencias:
        ax.axhline(valor, color=cor, linestyle=estilo, linewidth=1.4, zorder=1,
                   label=f"{rotulo} = {fmt(valor)}")
    valores = [*y.values(), *(r[0] for r in referencias)]
    ax.set_ylim(min(valores) - 0.03, max(valores) + 0.05)
    ax.set_xlim(-8, trajetoria_json["chamadas_metrica"] + 4)
    ax.set_xlabel("Chamadas de metrica")
    ax.set_ylabel("F1-score medio na validacao")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6,
              fontsize="small", columnspacing=1.0)
    salvar(fig, "fig1_protegi", saida_dir)


def fig_f1(resultados: dict, anotador: dict | None, bracos: list, saida_dir: Path) -> None:
    bracos = [b for b in bracos if b[0] in resultados]
    fig, eixos = plt.subplots(1, 2, figsize=(COLUNA, 3.4), sharey=True)
    for ax, (chave, titulo) in zip(eixos, (("deteccao", "Deteccao"),
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
    salvar(fig, "fig2_f1", saida_dir)


def fig_kappa(resultados: dict, anotador: dict | None, bracos: list, saida_dir: Path) -> None:
    bracos = [b for b in bracos if b[0] in resultados]
    n = len(bracos)
    largura = 0.8 / n
    fig, ax = plt.subplots(figsize=(COLUNA, 3.6))
    for j, (b, _, rotulo) in enumerate(bracos):
        for i, campo in enumerate(CATEGORICOS):
            k = resultados[b]["kappa"].get(campo) or 0.0
            xi = i - 0.4 + largura * (j + 0.5)
            ax.bar(xi, k, width=largura, color=COR[b], edgecolor=SUPERFICIE, linewidth=1,
                   label=rotulo if i == 0 else None)
            ks = [s["kappa"].get(campo) for s in resultados[b]["por_semente"]
                  if s["kappa"].get(campo) is not None]
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
    salvar(fig, "fig3_kappa", saida_dir)


def fig_confusao(resultados: dict, bracos: list, saida_dir: Path) -> None:
    bracos = [b for b in bracos if b[0] in resultados]
    linhas = METRICAS + ["ausente"]
    colunas = METRICAS + ["ausente"]
    rotulo_linha = [ROTULO_METRICA[m] for m in METRICAS] + ["(fora do gold)"]
    rotulo_coluna = [ROTULO_METRICA[m] for m in METRICAS] + ["(nao achou)"]
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
    salvar(fig, "fig4_confusao", saida_dir)


def fig_val_teste(val: dict[str, float], resultados: dict, bracos: list, saida_dir: Path) -> None:
    bracos = [b for b in bracos if b[0] in resultados and b[0] in val]
    fig, ax = plt.subplots(figsize=(COLUNA * 0.75, 3.6))
    for b, _, rotulo in bracos:
        v, t = val[b], resultados[b]["nota"]
        ax.plot([0, 1], [v, t], color=COR[b], linewidth=2, marker=MARCADOR[b], markersize=7,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.2,
                label=f"{rotulo}: {fmt(v)} $\\to$ {fmt(t)}")
    ax.set_xticks([0, 1], ["Validacao", "Teste"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("F1-score medio")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), handlelength=1.6)
    salvar(fig, "fig5_val_teste", saida_dir)


# ---------------------------------------------------------------------
# Tabela e main
# ---------------------------------------------------------------------


def linha_tex(nome: str, r: dict) -> str:
    def f1(chave):
        lo, hi = r[f"ic_{chave}"]
        return f"{r[chave]['f1']:.2f} [{lo:.2f}; {hi:.2f}]".replace(".", ",")
    ks = " & ".join("--" if not r["kappa"] or r["kappa"].get(c) is None
                    else f"{r['kappa'][c]:.2f}".replace(".", ",")
                    for c in CATEGORICOS)
    return f"{nome} & {f1('deteccao')} & {f1('tupla')} & {r['nota']:.2f}".replace(".", ",") \
        + f" & {ks} \\\\"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="gold_final.jsonl (doc_id, extracoes, particao).")
    ap.add_argument("--output-dir", required=True, dest="output_dir",
                     help="Pasta 'results/' do rodar_bracos.py (contem test/ e val/).")
    ap.add_argument("--braco-c", default="c_protegi", dest="braco_c",
                     help="Rotulo do braco C usado nos arquivos <rotulo>_s<seed>.jsonl "
                          "(padrao: c_protegi, igual ao exemplo do rodar_bracos.py).")
    ap.add_argument("--candidatos-json", default=None, dest="candidatos_json",
                     help="Opcional: candidatos.json do ProTeGi (trajetoria da busca). "
                          "Sem isso, fig1 e fig5 sao puladas.")
    ap.add_argument("--gold-piloto-dir", default=None, dest="gold_piloto_dir",
                     help="Opcional: pasta com piloto_r1.jsonl/piloto_r2.jsonl "
                          "(estudo de concordancia entre anotadores).")
    ap.add_argument("--saida", required=True, help="Pasta onde gravar figuras/tabela/analise.json.")
    args = ap.parse_args()

    saida_dir = Path(args.saida)
    saida_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.output_dir)
    gold_path = Path(args.gold)

    bracos = [("A", ROTULO_ARQUIVO_A, "A"), ("C", args.braco_c, "C")]

    gold_teste = carregar_gold_particao(gold_path, "test")
    gold_val = carregar_gold_particao(gold_path, "val")

    resultados = {}
    for chave, rotulo_arquivo, _ in bracos:
        r = avaliar_braco(gold_teste, results_dir, rotulo_arquivo)
        if r:
            resultados[chave] = r
            print(f"{chave:<3} {r['sementes']} sementes  deteccao {r['deteccao']['f1']:.3f}  "
                  f"tupla {r['tupla']['f1']:.3f}  nota {r['nota']:.3f}")
        else:
            print(f"{chave:<3} sem arquivos de teste encontrados para '{rotulo_arquivo}' em "
                  f"{results_dir / 'test'} -- pulando.")

    gold_piloto_dir = Path(args.gold_piloto_dir) if args.gold_piloto_dir else None
    anotador = avaliar_anotador(gold_piloto_dir)

    trajetoria = None
    if args.candidatos_json:
        caminho_traj = Path(args.candidatos_json)
        if caminho_traj.exists():
            trajetoria = json.loads(caminho_traj.read_text(encoding="utf-8"))
        else:
            print(f"[aviso] --candidatos-json {caminho_traj} nao existe -- pulando fig1/fig5.")

    val_c = nota_val_braco_c(results_dir, args.braco_c, gold_val) if gold_val else None

    permutacoes = {}
    chaves = [b for b, _, _ in bracos if b in resultados]
    for i, x in enumerate(chaves):
        for y in chaves[i + 1:]:
            permutacoes[f"{y} - {x}"] = {
                m: teste_permutacao(resultados[y]["_contagens"], resultados[x]["_contagens"], m)
                for m in ("deteccao", "tupla")}

    if trajetoria:
        fig_protegi(trajetoria, anotador, bracos, saida_dir)
    if resultados:
        fig_f1(resultados, anotador, bracos, saida_dir)
        if TEM_CONCORDANCIA:
            fig_kappa(resultados, anotador, bracos, saida_dir)
        fig_confusao(resultados, bracos, saida_dir)
    if trajetoria and resultados and val_c is not None:
        val = {"C": val_c}
        if trajetoria["candidatas"]:
            indice_inicial = next((c["indice"] for c in trajetoria["candidatas"]
                                   if not c["pais"]), None)
            if indice_inicial is not None:
                y0 = next(c["nota_val"] for c in trajetoria["candidatas"]
                         if c["indice"] == indice_inicial)
                val["A"] = y0
        fig_val_teste(val, resultados, bracos, saida_dir)

    linhas = [linha_tex(b, resultados[b]) for b in chaves]
    if anotador:
        linhas.append(linha_tex("reteste*", anotador))
    (saida_dir / "tabela_teste.tex").write_text("\n".join(linhas) + "\n", encoding="ascii")

    publico = {b: {k: v for k, v in r.items() if not k.startswith("_")}
               for b, r in resultados.items()}
    (saida_dir / "analise.json").write_text(json.dumps(
        {"teste": publico, "anotador": anotador, "permutacao": permutacoes,
         "val_c_protegi": val_c}, indent=1, ensure_ascii=True), encoding="ascii")

    for par, m in permutacoes.items():
        print(f"{par:<8} " + "  ".join(f"{k} dif {v['diferenca']:+.3f} p {v['p_valor']:.3f}"
                                       for k, v in m.items()))
    print(f"\nsaidas em {saida_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())