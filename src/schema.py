"""Schema de extracao (F2-02).

Contrato unico da anotacao, da saida do LLM e da avaliacao. Traduz os campos
do documento de escopo, com dois ajustes decididos no projeto:

- `unit` foi acrescentado. Sem ele, `value = 0.3` e ambiguo entre 0,3 % e
  0,3 ppm, tres ordens de grandeza de diferenca.
- `entity_type` e derivado de `target_entity` pelo lexico, nao anotado. O
  anotador so corrige quando a derivacao erra.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MetricType = Literal[
    "Bulk Concentration",
    "Individual Entity Concentration",
    "Process Metric",
    "Invalid Candidate",
]

EntityType = Literal["element", "element_group", "oxide", "mineral", "other"]

# Vocabulario fechado (decisao do plano de 14 dias). Sem campo livre, sem
# decisao no calor do momento.
ContextModifier = Literal[
    "none", "average", "approximate", "greater_than", "less_than", "up_to"
]

# Unidades aceitas. As de solucao implicam Process Metric e nunca sao
# convertidas para teor de solido sem densidade.
UNIDADES_MASSA = ("%", "wt%", "ppm", "ppb", "g/t", "mg/kg", "kg/t", "ug/g", "permil")
UNIDADES_SOLUCAO = ("mg/L", "g/L", "mol/L", "M")
UNIDADES = (*UNIDADES_MASSA, *UNIDADES_SOLUCAO, "dimensionless")

# Normalizacao de grafias que aparecem no texto para a forma canonica acima.
_ALIAS_UNIDADE = {
    "wt.%": "wt%", "wt %": "wt%", "wt. %": "wt%", "%wt": "wt%",
    "weight%": "wt%", "wt-%": "wt%",
    "μg/g": "ug/g", "µg/g": "ug/g", "mcg/g": "ug/g",
    "mg/l": "mg/L", "g/l": "g/L", "mol/l": "mol/L",
    "‰": "permil", "per mil": "permil", "ppt": "permil",
    "g/tonne": "g/t", "gpt": "g/t", "ppmw": "ppm",
}


def normalizar_unidade(bruta: str | None) -> str | None:
    """Converte a grafia do texto para a forma canonica do schema."""
    if bruta is None:
        return None
    chave = re.sub(r"\s+", "", bruta.strip()).lower()
    chave_com_espaco = bruta.strip().lower()
    for candidata in (chave, chave_com_espaco):
        if candidata in _ALIAS_UNIDADE:
            return _ALIAS_UNIDADE[candidata]
    for unidade in UNIDADES:
        if candidata_igual(chave, unidade):
            return unidade
    return None


def candidata_igual(chave: str, unidade: str) -> bool:
    return chave == re.sub(r"\s+", "", unidade).lower()


class Extracao(BaseModel):
    """Um valor quantitativo extraido de um resumo."""

    value: float | None = Field(
        description="Valor numerico. Limite inferior quando is_range e verdadeiro."
    )
    value_max: float | None = Field(
        default=None, description="Limite superior. Preenchido apenas em intervalo."
    )
    is_range: bool = Field(description="O dado corresponde a um intervalo?")
    unit: str | None = Field(default=None, description="Unidade canonica do valor.")
    sentence: str = Field(
        min_length=1,
        description="Trecho literal do resumo onde o valor aparece. Serve de ancora.",
    )
    metric_type: MetricType
    target_entity: str = Field(
        min_length=1, description='Entidade a que o valor se refere: "TREO", "Nd", etc.'
    )
    entity_type: EntityType = Field(
        default="other", description="Derivado de target_entity pelo lexico."
    )
    context_modifier: ContextModifier = "none"

    @field_validator("unit")
    @classmethod
    def _unidade_canonica(cls, v: str | None) -> str | None:
        if v is None:
            return None
        canonica = normalizar_unidade(v)
        if canonica is None:
            raise ValueError(f"unidade nao reconhecida: {v!r}")
        return canonica

    @field_validator("sentence", "target_entity")
    @classmethod
    def _sem_espaco_sobrando(cls, v: str) -> str:
        return re.sub(r"\s+", " ", v).strip()

    @model_validator(mode="after")
    def _coerencia(self) -> Extracao:
        if self.metric_type == "Invalid Candidate":
            return self

        if self.is_range:
            if self.value_max is None:
                raise ValueError("is_range verdadeiro exige value_max")
            if self.value is not None and self.value_max < self.value:
                raise ValueError(
                    f"intervalo invertido: {self.value} > {self.value_max}"
                )
        elif self.value_max is not None:
            raise ValueError("value_max preenchido em valor unico")

        if self.value is None and self.value_max is None:
            raise ValueError("extracao sem nenhum valor numerico")

        if self.unit in UNIDADES_SOLUCAO and self.metric_type != "Process Metric":
            raise ValueError(
                f"unidade de solucao ({self.unit}) exige metric_type Process Metric"
            )
        return self

    @property
    def e_solucao(self) -> bool:
        return self.unit in UNIDADES_SOLUCAO


class Registro(BaseModel):
    """Todas as extracoes de um documento.

    Lista vazia e resultado valido e conta na avaliacao: resumo sem nenhum
    valor quantitativo de terras raras e o caso mais comum do corpus.
    """

    doc_id: str
    extracoes: list[Extracao] = Field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.extracoes)
