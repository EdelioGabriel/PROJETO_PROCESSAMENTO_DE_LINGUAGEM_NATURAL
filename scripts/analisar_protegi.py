"""Orquestra os bracos do experimento (A/B/C) sobre TESTE, com 3 sementes cada,
usando o pipeline COMPLETO de extracao do projeto (validacao Pydantic +
ancoragem + retentativa, em src.extrair.extrair), nao a extracao simples do
apo.py -- para que a comparacao entre bracos (A ingenuo, C ProTeGi, C GEPA)
nao seja distorcida por diferencas de pipeline de extracao.

Reaproveita, sem duplicar:
  - apo.py: carregamento e split de dados (gold_final.jsonl por particao,
    candidatos.jsonl, TERRAS_RARAS.csv como fonte de texto secundaria) --
    especifico deste projeto, nao existe no codigo do colaborador.
  - src.extrair.extrair / src.llm.processar_lote / src.entidades.Lexico /
    src.avaliar.avaliar+imprimir: extracao com validacao+ancoragem+retry,
    execucao em lote retomavel, e avaliacao com IC por bootstrap -- do
    colaborador, usados como estao.

Sementes (SEMENTES = 0, 1, 2): a instalacao da IlumA rejeita temperatura
abaixo de 0.5, entao nenhuma chamada e deterministica -- variancia se mede
por REPETICAO, nao por um parametro "seed" da API (src.llm nem manda um
"seed" pro servidor). Para que 3 repeticoes nao batam na MESMA resposta
cacheada -- a chave de cache usa (mensagens, modelo, temperatura, max_tokens,
run, extras), e o "run" interno de extrair() sempre recomeca do zero a cada
chamada -- cada semente usa um `dir_cache` proprio (cache/sementes/s<seed>/):
e o unico gancho que extrair() expoe pra isolar repeticoes de fora.

Convencao de saida esperada por analisar.py:
    <output-dir>/test/<rotulo>_s<seed>.jsonl   para seed em 0, 1, 2
    <output-dir>/val/<rotulo>_s0.jsonl         so para bracos com --tambem-validacao

`rotulo` e o nome usado em BRACOS/RESULTADOS no analisar.py (ex.
"v00_ingenuo", "b_regras", "c_protegi"). A e C normalmente NAO precisam de
--tambem-validacao: a nota de validacao deles ja vem de `candidatos.json`
(campo "nota_val"), gerado pelo `otimizar` do apo.py.

Cada chamada e retomavel: `processar_lote` (src.llm) pula doc_id ja
presentes no .jsonl de saida -- interromper e rodar de novo continua de
onde parou.

Layout de projeto esperado (RAIZ = pasta acima de scripts/):
    RAIZ/scripts/apo.py
    RAIZ/scripts/rodar_bracos.py   (este arquivo)
    RAIZ/src/{config,llm,entidades,extrair,avaliar,critica,schema}.py
    RAIZ/data/lexico_ree.yaml
    RAIZ/config/config.yaml        (secao "llm")
    RAIZ/.env                      (ILUMA_TOKEN=...)

Uso tipico:

    python rodar_bracos.py --gold ../gold_standard/gold_final.jsonl --candidatos ../data/candidatos.jsonl --resumos-extra ../data/TERRAS_RARAS.csv --output-dir ../outputs/output_apo_oficial_hpc/results --braco v00_ingenuo=../prompts/v00_ingenuo.txt --braco c_protegi=../outputs/output_apo_oficial_hpc/prompt_final.txt

Com o Braco B tambem na validacao:

    python rodar_bracos.py \\
        ... \\
        --braco b_regras=../prompts/b_regras.txt \\
        --tambem-validacao b_regras
"""

import argparse
import sys
from pathlib import Path

# apo.py (mesma pasta deste script) -- so para carregamento/split de dados.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apo  # noqa: E402

# src/ do colaborador (um nivel acima de scripts/).
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.avaliar import avaliar as avaliar_src, imprimir as imprimir_src  # noqa: E402
from src.entidades import Lexico  # noqa: E402
from src.extrair import carregar_resultados_llm, extrair  # noqa: E402
from src.llm import processar_lote  # noqa: E402
from src.schema import Registro  # noqa: E402

SEMENTES = (0, 1, 2)


def _para_registros(gold_bruto: dict) -> dict:
    """{doc_id: [extracoes...]} (formato do apo.py) -> {doc_id: Registro}
    (formato Pydantic que src.avaliar/src.extrair esperam). Pode levantar
    ValidationError se o gold tiver uma extracao fora do schema -- nesse
    caso o problema e no gold_final.jsonl, nao neste script."""
    return {d: Registro(doc_id=d, extracoes=ex) for d, ex in gold_bruto.items()}


def _fabrica_processar(prompt_texto: str, lex: Lexico, cache_dir: Path):
    def processar(item: tuple) -> dict:
        doc_id, resumo = item
        resultado = extrair(resumo, doc_id, prompt_texto, lex=lex, dir_cache=cache_dir)
        return resultado.para_json()
    return processar


def _rodar_e_avaliar(prompt_texto: str, lex: Lexico, abstracts: dict, gold: dict,
                      saida: Path, cache_dir: Path, rotulo_print: str) -> None:
    print(f"\n=== {rotulo_print} -> {saida} ===")
    itens = list(abstracts.items())
    processar_lote(
        itens, _fabrica_processar(prompt_texto, lex, cache_dir), saida,
        id_de=lambda item: item[0], descricao=rotulo_print,
    )
    predicoes = carregar_resultados_llm(saida)
    resultado = avaliar_src(gold, predicoes)
    imprimir_src(resultado, titulo=rotulo_print)


def rodar_braco_teste(rotulo: str, prompt_path: Path, abstracts_teste: dict,
                       gold_teste: dict, saida_dir: Path, lex: Lexico) -> None:
    prompt_texto = prompt_path.read_text(encoding="utf-8")
    for s in SEMENTES:
        saida = saida_dir / "test" / f"{rotulo}_s{s}.jsonl"
        cache_dir = RAIZ / "cache" / "sementes" / f"s{s}"
        _rodar_e_avaliar(prompt_texto, lex, abstracts_teste, gold_teste, saida, cache_dir,
                          rotulo_print=f"{rotulo} | teste | seed={s}")


def rodar_braco_validacao(rotulo: str, prompt_path: Path, abstracts_val: dict,
                           gold_val: dict, saida_dir: Path, lex: Lexico) -> None:
    prompt_texto = prompt_path.read_text(encoding="utf-8")
    saida = saida_dir / "val" / f"{rotulo}_s0.jsonl"
    cache_dir = RAIZ / "cache" / "sementes" / "s0"
    _rodar_e_avaliar(prompt_texto, lex, abstracts_val, gold_val, saida, cache_dir,
                      rotulo_print=f"{rotulo} | validacao (1 semente, sem replicacao)")


def _parse_bracos(pares: list) -> dict:
    bracos = {}
    for item in pares:
        if "=" not in item:
            raise SystemExit(f"--braco invalido: {item!r} (esperado rotulo=caminho_do_prompt.txt)")
        rotulo, caminho = item.split("=", 1)
        bracos[rotulo] = Path(caminho)
    return bracos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="Extracoes anotadas manualmente (doc_id, extracoes, particao).")
    ap.add_argument("--candidatos", required=True, help="Corpus completo (doc_id, titulo, resumo).")
    ap.add_argument("--resumos-extra", default=None, dest="resumos_extra",
                     help="Opcional: CSV (ex. TERRAS_RARAS.csv) usado como fonte de texto secundaria.")
    ap.add_argument("--output-dir", required=True, dest="output_dir",
                     help="Pasta 'results/' -- os .jsonl vao em <output-dir>/test/ e <output-dir>/val/.")
    ap.add_argument("--braco", action="append", required=True, dest="bracos",
                     metavar="ROTULO=CAMINHO",
                     help="rotulo=caminho_do_prompt.txt, repetivel.")
    ap.add_argument("--tambem-validacao", action="append", default=[], dest="tambem_validacao",
                     metavar="ROTULO",
                     help="Rotulo(s) que tambem devem rodar 1x (sem replicacao) na validacao "
                          "-- tipicamente so o Braco B. Repetivel.")
    args = ap.parse_args()

    bracos = _parse_bracos(args.bracos)
    for rotulo in args.tambem_validacao:
        if rotulo not in bracos:
            raise SystemExit(f"--tambem-validacao {rotulo!r} nao esta entre os --braco passados: "
                              f"{sorted(bracos)}")

    print("Carregando lexico (src.entidades.Lexico)...")
    lex = Lexico.carregar()

    gold = apo.carregar_gold(Path(args.gold))
    candidatos = apo.carregar_candidatos(Path(args.candidatos))
    resumos_extra = apo.carregar_resumos_csv(Path(args.resumos_extra)) if args.resumos_extra else {}
    _, gold_val_bruto, gold_teste_bruto = apo.dividir_gold_por_particao(gold)
    abstracts_teste = apo.textos_para_ids(gold_teste_bruto.keys(), candidatos, resumos_extra)
    abstracts_val = apo.textos_para_ids(gold_val_bruto.keys(), candidatos, resumos_extra)

    print("Convertendo gold para Registro (validacao Pydantic)...")
    gold_teste = _para_registros(gold_teste_bruto)
    gold_val = _para_registros(gold_val_bruto)

    saida_dir = Path(args.output_dir)
    print(f"\nBracos a rodar: {list(bracos)} | teste={len(abstracts_teste)} abstracts, "
          f"{len(SEMENTES)} semente(s) cada | validacao (so {args.tambem_validacao or 'nenhum'}): "
          f"{len(abstracts_val)} abstracts, 1 semente")

    for rotulo, prompt_path in bracos.items():
        if not prompt_path.exists():
            raise SystemExit(f"--braco {rotulo}: arquivo nao encontrado: {prompt_path}")
        rodar_braco_teste(rotulo, prompt_path, abstracts_teste, gold_teste, saida_dir, lex)
        if rotulo in args.tambem_validacao:
            rodar_braco_validacao(rotulo, prompt_path, abstracts_val, gold_val, saida_dir, lex)

    print(f"\nConcluido. Resultados em: {saida_dir}")


if __name__ == "__main__":
    main()