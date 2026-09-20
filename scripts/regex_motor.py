"""
Script para realizar busca de valores via Regex e API da Iluma

Autor: Edélio Gabriel M. de Jesus

Exemplo de chamada:

python regex_motor.py --input ..\outputs\TERRAS_RARAS.csv --output ..\outputs\abstracts_candidatos.csv   
             
"""

import argparse
from pathlib import Path

import regex as re
import pandas as pd


class FiltroTerrasRaras:
    """Peneira regex: decide quais abstracts merecem processamento mais caro depois."""

    ELEMENT_PATTERN = (
        r"\b(?:La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Sc|Y|"
        r"lanthanum|cerium|praseodymium|neodymium|promethium|samarium|"
        r"europium|gadolinium|terbium|dysprosium|holmium|erbium|thulium|"
        r"ytterbium|lutetium|scandium|yttrium)\b"
    )

    PROPRIEDADE_PATTERN = (
        r"(?:\b(?:TREO(?:\s*\+\s*Y)?|REO|REE)\b|"
        r"\btotal\s+rare[-\s]?earth\s+oxides?\b|"
        r"\b(?:rare[-\s]+earth(?:\s+elements?|\s+oxides?))\s+"
        r"(?:composition|content|concentration|grade|distribution|"
        r"abundance|proportion|ratio)\b|"
        r"\b(?:composition|content|concentration|grade|distribution|"
        r"abundance|proportion|ratio)\s+of\s+(?:the\s+)?"
        r"(?:rare[-\s]+earth(?:\s+elements?|\s+oxides?)|REE|REO)\b)"
    )

    VALOR_PATTERN = (
        r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
        r"\s*(?:wt\s*%|%|ppm|g/t)\b"
    )

    def __init__(self, caminho_entrada: str, caminho_saida: str = "abstracts_candidatos.csv"):
        self.caminho_entrada = Path(caminho_entrada)
        self.caminho_saida = Path(caminho_saida)

        self.propriedade = re.compile(self.PROPRIEDADE_PATTERN, flags=re.IGNORECASE)
        self.valor = re.compile(self.VALOR_PATTERN, flags=re.IGNORECASE)
        self.elemento = re.compile(self.ELEMENT_PATTERN, flags=re.IGNORECASE)

        self.dados_total = None
        self.abstracts_filtrados = None

    def carregar(self) -> None:
        """Lê o CSV de entrada."""
        self.dados_total = pd.read_csv(self.caminho_entrada)

    def vale_a_chamada(self, texto: str) -> bool:
        """O resumo tem chance de conter o que procuramos?"""
        return bool(self.propriedade.search(texto) and self.valor.search(texto))

    def filtrar(self) -> pd.DataFrame:
        """Aplica a peneira e monta o DataFrame de candidatos."""
        if self.dados_total is None:
            self.carregar()

        abstracts_ids = []
        abstracts_candidatos = []

        for i, abstract in enumerate(self.dados_total['abstract']):
            if pd.isna(abstract):
                continue

            texto = str(abstract)
            if self.vale_a_chamada(texto):
                abstracts_ids.append(i)
                abstracts_candidatos.append(texto)

        self.abstracts_filtrados = pd.DataFrame({
            'Abstract_id': abstracts_ids,
            'Abstract': abstracts_candidatos,
        })
        return self.abstracts_filtrados

    def salvar(self) -> None:
        """Salva os candidatos filtrados no caminho de saída."""
        if self.abstracts_filtrados is None:
            raise RuntimeError("Nada para salvar: chame filtrar() antes de salvar().")

        self.caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        self.abstracts_filtrados.to_csv(self.caminho_saida, index=False)

    def relatorio(self) -> str:
        """Monta o texto de resumo (resumos -> candidatos, chamadas economizadas)."""
        total_validos = self.dados_total['abstract'].notna().sum()
        n_candidatos = len(self.abstracts_filtrados)
        linhas = [
            f"{total_validos} resumos → {n_candidatos} candidatos "
            f"({n_candidatos / total_validos:.1%})",
            f"chamadas economizadas: {total_validos - n_candidatos}",
        ]
        return "\n".join(linhas)

    def executar(self) -> None:
        """Roda o pipeline completo: carregar -> filtrar -> salvar -> relatar."""
        self.carregar()
        self.filtrar()
        self.salvar()
        print(self.relatorio())


def main():
    parser = argparse.ArgumentParser(
        description="Filtra abstracts com dados quantitativos de terras raras via regex."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Caminho do CSV de entrada (deve conter a coluna 'abstract').",
    )
    parser.add_argument(
        "--output",
        default="abstracts_candidatos.csv",
        help="Caminho do CSV de saída (padrão: ./abstracts_candidatos.csv).",
    )
    args = parser.parse_args()

    filtro = FiltroTerrasRaras(args.input, args.output)
    filtro.executar()


if __name__ == "__main__":
    main()