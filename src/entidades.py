"""Lexico de terras raras e derivacao de `entity_type` (F1-02).

O `entity_type` do schema nao e anotado a mao: e derivado de `target_entity`
pelo lexico, que ja classifica cada termo. A derivacao e deterministica e
auditavel, e o anotador so corrige quando ela erra.

    from src.entidades import Lexico
    lex = Lexico.carregar()
    lex.tipo_de("Nd2O3")        # -> "oxide"
    lex.tipo_de("bastnasite")   # -> "mineral"
    lex.canonico("TREO")        # -> "TREO"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

from src.config import RAIZ

CAMINHO_PADRAO = RAIZ / "data" / "lexico_ree.yaml"

# Secoes do YAML que descrevem entidades alvo, com o entity_type de cada uma.
SECOES_ENTIDADE = ("elementos", "grupos", "oxidos", "minerais", "materiais")

TIPOS_VALIDOS = ("element", "element_group", "oxide", "mineral", "other")


@dataclass
class Lexico:
    """Indice de termos do dominio, com busca exata e aproximada."""

    bruto: dict
    # forma minuscula -> (canonico, entity_type, grafia original)
    indice: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    sensivel_a_caixa: set[str] = field(default_factory=set)
    padrao_termos: re.Pattern | None = None
    padrao_processo: re.Pattern | None = None

    @classmethod
    def carregar(cls, caminho: Path | str | None = None) -> Lexico:
        caminho = Path(caminho) if caminho else CAMINHO_PADRAO
        bruto = yaml.safe_load(caminho.read_text(encoding="utf-8"))
        lex = cls(bruto=bruto, sensivel_a_caixa=set(bruto.get("sensivel_a_caixa", [])))
        lex._construir_indice()
        lex._construir_padroes()
        return lex

    def _construir_indice(self) -> None:
        for secao in SECOES_ENTIDADE:
            tipo = self.bruto[secao]["entity_type"]
            for termo in self.bruto[secao]["termos"]:
                canonico = termo["canonico"]
                formas = {canonico, termo.get("nome", ""), *termo.get("variantes", [])}
                for forma in filter(None, formas):
                    self.indice[forma.lower()] = (canonico, tipo, forma)

    def _construir_padroes(self) -> None:
        # Siglas sensiveis a caixa entram num ramo separado do padrao. Em
        # minuscula, "ree" casaria com o sobrenome Rees e com "deg ree".
        sensiveis, insensiveis = [], []
        for _, _, original in self.indice.values():
            alvo = sensiveis if original in self.sensivel_a_caixa else insensiveis
            alvo.append(re.escape(original))

        partes = []
        if insensiveis:
            corpo = "|".join(sorted(set(insensiveis), key=len, reverse=True))
            partes.append(rf"(?i:\b(?:{corpo})\b)")
        if sensiveis:
            corpo = "|".join(sorted(set(sensiveis), key=len, reverse=True))
            partes.append(rf"\b(?:{corpo})\b")
        self.padrao_termos = re.compile("|".join(partes))

        processo = [re.escape(t) for t in self.bruto["processo"]["termos"]]
        corpo = "|".join(sorted(processo, key=len, reverse=True))
        self.padrao_processo = re.compile(rf"(?i:\b(?:{corpo}))")

    # --- consulta --------------------------------------------------------

    def tipo_de(self, entidade: str, limiar_fuzzy: int = 90) -> str:
        """Devolve o `entity_type` de uma `target_entity`.

        Tenta casamento exato, depois aproximado. Sem casamento, devolve
        "other", que e o valor do escopo para os casos nao classificaveis.
        """
        achado = self._buscar(entidade, limiar_fuzzy)
        return achado[1] if achado else "other"

    def canonico(self, entidade: str, limiar_fuzzy: int = 90) -> str | None:
        """Forma canonica da entidade, ou None se nao reconhecida."""
        achado = self._buscar(entidade, limiar_fuzzy)
        return achado[0] if achado else None

    def _buscar(self, entidade: str, limiar_fuzzy: int) -> tuple[str, str, str] | None:
        if not entidade:
            return None
        chave = entidade.strip().lower()
        if chave in self.indice:
            return self.indice[chave]
        # "REE 2 O 3" (subscrito perdido na extracao do texto) -> "ree2o3"
        compacta = re.sub(r"\s+", "", chave)
        if compacta in self.indice:
            return self.indice[compacta]
        if len(chave) < 4:
            return None
        aproximado = rf_process.extractOne(
            chave, list(self.indice.keys()), scorer=fuzz.ratio, score_cutoff=limiar_fuzzy
        )
        return self.indice[aproximado[0]] if aproximado else None

    def termos_em(self, texto: str) -> list[str]:
        """Termos do lexico presentes no texto, na ordem de ocorrencia."""
        return self.padrao_termos.findall(texto) if self.padrao_termos else []

    def tem_termo(self, texto: str) -> bool:
        return bool(self.padrao_termos and self.padrao_termos.search(texto))

    def tem_processo(self, texto: str) -> bool:
        return bool(self.padrao_processo and self.padrao_processo.search(texto))

    @property
    def unidades_massa(self) -> list[str]:
        return self.bruto["unidades"]["massa_por_massa"]

    @property
    def unidades_solucao(self) -> list[str]:
        return self.bruto["unidades"]["solucao"]