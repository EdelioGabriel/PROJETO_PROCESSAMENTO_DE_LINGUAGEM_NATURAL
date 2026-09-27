"""Orquestra os bracos do experimento (A/B/C) sobre TESTE, com 3 sementes cada.

Script FINO: `import apo` e reaproveita as funcoes de la (get_client,
carregar_config, carregar_gold, carregar_candidatos, carregar_resumos_csv,
dividir_gold_por_particao, textos_para_ids, rodar_prompt_sobre_abstracts,
carregar_jsonl_por_doc, avaliar, _imprimir_resultado) -- NENHUMA logica de
extracao, parsing ou avaliacao e duplicada aqui. Este arquivo so decide QUEM
roda QUANDO, na convencao de arquivos que `analisar.py` espera:

    <output-dir>/test/<rotulo>_s<seed>.jsonl   para seed em 0, 1, 2
    <output-dir>/val/<rotulo>_s0.jsonl         so para bracos com --tambem-validacao

`rotulo` e o nome usado em BRACOS/RESULTADOS no analisar.py (ex.
"v00_ingenuo", "b_regras", "c_protegi"). A e C normalmente NAO precisam de
--tambem-validacao: a nota de validacao deles ja vem de `candidatos.json`
(campo "nota_val" de cada candidato), gerado pelo `otimizar` -- rodar de novo
so pra validacao seria custo de API redundante. B, se nao passar pelo
`otimizar`, tipicamente precisa de --tambem-validacao (1 semente, sem
replicacao, so pra ter uma nota de referencia no grafico).

Cada chamada e retomavel: como `apo.rodar_prompt_sobre_abstracts` pula doc_id
ja presentes no .jsonl de saida, interromper e rodar de novo (mesmo comando)
continua de onde parou, sem duplicar nem perder trabalho.

Uso tipico:

    python rodar_bracos.py \\
        --gold ../gold_standard/gold_final.jsonl \\
        --candidatos ../data/candidatos.jsonl \\
        --resumos-extra ../data/TERRAS_RARAS.csv \\
        --config ../config/config.yaml \\
        --output-dir ../outputs/results \\
        --braco v00_ingenuo=../outputs/output_apo_oficial/v00_ingenuo.txt \\
        --braco c_protegi=../outputs/output_apo_oficial/prompt_final.txt

Com o Braco B (baseado em regras), rodando tambem 1x na validacao:

    python rodar_bracos.py \\
        --gold ../gold_standard/gold_final.jsonl \\
        --candidatos ../data/candidatos.jsonl \\
        --resumos-extra ../data/TERRAS_RARAS.csv \\
        --config ../config/config.yaml \\
        --output-dir ../outputs/results \\
        --braco v00_ingenuo=../outputs/output_apo_oficial/v00_ingenuo.txt \\
        --braco b_regras=../prompts/b_regras.txt \\
        --braco c_protegi=../outputs/output_apo_oficial/prompt_final.txt \\
        --tambem-validacao b_regras
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apo  # reaproveita tudo -- ver docstring acima

SEMENTES = (0, 1, 2)


def _rodar_e_avaliar(client, prompt_texto: str, abstracts: dict, gold: dict,
                      saida: Path, seed: int, rotulo_print: str) -> None:
    print(f"\n=== {rotulo_print} -> {saida} ===")
    apo.rodar_prompt_sobre_abstracts(client, prompt_texto, abstracts, saida, seed=seed)
    predicoes = apo.carregar_jsonl_por_doc(saida)
    resultado = apo.avaliar(predicoes, gold)
    apo._imprimir_resultado(rotulo_print, resultado)


def rodar_braco_teste(client, rotulo: str, prompt_path: Path,
                       abstracts_teste: dict, gold_teste: dict, saida_dir: Path) -> None:
    prompt_texto = prompt_path.read_text(encoding="utf-8")
    for s in SEMENTES:
        saida = saida_dir / "test" / f"{rotulo}_s{s}.jsonl"
        _rodar_e_avaliar(client, prompt_texto, abstracts_teste, gold_teste, saida, s,
                          rotulo_print=f"{rotulo} | teste | seed={s}")


def rodar_braco_validacao(client, rotulo: str, prompt_path: Path,
                           abstracts_val: dict, gold_val: dict, saida_dir: Path) -> None:
    prompt_texto = prompt_path.read_text(encoding="utf-8")
    saida = saida_dir / "val" / f"{rotulo}_s0.jsonl"
    _rodar_e_avaliar(client, prompt_texto, abstracts_val, gold_val, saida, 0,
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
    ap.add_argument("--config", default=str(apo.CONFIG_PADRAO), help="config.yaml (mesmo do apo.py).")
    ap.add_argument("--output-dir", required=True, dest="output_dir",
                     help="Pasta 'results/' -- os .jsonl vao em <output-dir>/test/ e <output-dir>/val/.")
    ap.add_argument("--braco", action="append", required=True, dest="bracos",
                     metavar="ROTULO=CAMINHO",
                     help="rotulo=caminho_do_prompt.txt, repetivel "
                          "(ex. --braco v00_ingenuo=... --braco c_protegi=...).")
    ap.add_argument("--tambem-validacao", action="append", default=[], dest="tambem_validacao",
                     metavar="ROTULO",
                     help="Rotulo(s) que tambem devem rodar 1x (sem replicacao) na validacao "
                          "-- tipicamente so o Braco B, ja que A/C tem 'nota_val' vinda do "
                          "candidatos.json do `otimizar`. Repetivel.")
    args = ap.parse_args()

    cfg = apo.carregar_config(Path(args.config))
    print(f"[config] modelo={cfg['modelo']} temperatura={cfg['temperatura']} "
          f"(piso {cfg['temperatura_piso']}) max_tokens={cfg['max_tokens']} "
          f"timeout={cfg['timeout_s']}s tentativas={cfg['tentativas']} "
          f"k={cfg['k_autoconsistencia']}")

    bracos = _parse_bracos(args.bracos)
    for rotulo in args.tambem_validacao:
        if rotulo not in bracos:
            raise SystemExit(f"--tambem-validacao {rotulo!r} nao esta entre os --braco passados: "
                              f"{sorted(bracos)}")

    client = apo.get_client()
    gold = apo.carregar_gold(Path(args.gold))
    candidatos = apo.carregar_candidatos(Path(args.candidatos))
    resumos_extra = apo.carregar_resumos_csv(Path(args.resumos_extra)) if args.resumos_extra else {}
    _, gold_val, gold_teste = apo.dividir_gold_por_particao(gold)
    abstracts_teste = apo.textos_para_ids(gold_teste.keys(), candidatos, resumos_extra)
    abstracts_val = apo.textos_para_ids(gold_val.keys(), candidatos, resumos_extra)

    saida_dir = Path(args.output_dir)
    print(f"\nBracos a rodar: {list(bracos)} | teste={len(abstracts_teste)} abstracts, "
          f"{len(SEMENTES)} semente(s) cada | validacao (so {args.tambem_validacao or 'nenhum'}): "
          f"{len(abstracts_val)} abstracts, 1 semente")

    for rotulo, prompt_path in bracos.items():
        if not prompt_path.exists():
            raise SystemExit(f"--braco {rotulo}: arquivo nao encontrado: {prompt_path}")
        rodar_braco_teste(client, rotulo, prompt_path, abstracts_teste, gold_teste, saida_dir)
        if rotulo in args.tambem_validacao:
            rodar_braco_validacao(client, rotulo, prompt_path, abstracts_val, gold_val, saida_dir)

    print(f"\nConcluido. Resultados em: {saida_dir}")


if __name__ == "__main__":
    main()