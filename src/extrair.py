"""Extracao com LLM: parser tolerante, validacao e ancoragem (F4-01 a F4-03).

Tres camadas, na ordem em que uma resposta do modelo as atravessa:

1. `ler_json` -- tolera cercas de markdown, texto solto em volta e lista nua
   no lugar do objeto. Padrao 3 do receituario da IlumA.
2. `validar` -- passa cada extracao pelo schema Pydantic. Extracoes invalidas
   sao separadas com o motivo, em vez de derrubarem o documento inteiro.
3. `verificar_ancoragem` -- rejeita extracao cuja `sentence` nao seja trecho
   literal do resumo, ou cujo `value` nao apareca dentro dessa sentenca.

A ancoragem e consequencia direta da decisao F(-1)-08: o resumo inteiro e a
entrada, e o campo `sentence` existe para tornar a extracao verificavel sem
julgamento humano. Um regex nunca devolve um numero que nao esta no texto; um
modelo pode, e com a mesma confianca do acerto.

A comparacao normaliza espacos em branco dos dois lados. A reconstrucao do
indice invertido do OpenAlex altera espacamento, e uma comparacao literal
rejeitaria ancoras corretas.

Sobre as re-tentativas: uma resposta pode falhar por dois motivos com
remedios opostos. Se veio mal formatada, reapresentar o erro ao modelo
resolve. Se foi cortada pelo limite de tokens (`finish_reason == "length"`),
pedir "responda so o JSON" nao adianta nada -- em modelos que raciocinam antes
de responder, o orcamento acaba no raciocinio e o `content` volta vazio. Nesse
caso a unica saida e repetir a pergunta com mais orcamento.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.config import RAIZ
from src.entidades import Lexico
from src.llm import config_llm, max_tokens_padrao, perguntar_completo
from src.schema import Extracao, Registro

PROMPTS = RAIZ / "prompts"

# Tentativas por documento, somando as duas causas de falha. Tres cobre um
# truncamento seguido de uma resposta mal formatada, ou dois truncamentos.
TENTATIVAS_PADRAO = 3

# Teto do orcamento de saida quando uma resposta e truncada e o orcamento
# dobra. Protege a cota contra um documento que nunca termina.
TETO_MAX_TOKENS_PADRAO = 16000

CERCA_MARKDOWN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
ESPACOS = re.compile(r"\s+")


class RespostaInvalida(ValueError):
    """A resposta do modelo nao pode ser lida como JSON."""


# ---------------------------------------------------------------------
# 1. Leitura tolerante
# ---------------------------------------------------------------------


def ler_json(texto: str) -> dict:
    """Extrai o JSON de uma resposta do modelo.

    Tolera cercas de markdown, frase antes ou depois do JSON, e lista nua no
    lugar do objeto `{"extracoes": [...]}`.
    """
    if texto is None:
        raise RespostaInvalida("resposta vazia")

    bruto = CERCA_MARKDOWN.sub("", texto.strip()).strip()

    for tentativa in (bruto, _maior_bloco(bruto, "{", "}"), _maior_bloco(bruto, "[", "]")):
        if not tentativa:
            continue
        try:
            dados = json.loads(tentativa)
        except json.JSONDecodeError:
            continue
        if isinstance(dados, list):
            return {"extracoes": dados}
        if isinstance(dados, dict):
            return dados

    raise RespostaInvalida(f"resposta nao e JSON: {bruto[:160]!r}")


def _maior_bloco(texto: str, abre: str, fecha: str) -> str | None:
    inicio, fim = texto.find(abre), texto.rfind(fecha)
    return texto[inicio : fim + 1] if 0 <= inicio < fim else None


def extrair_lista(dados: dict) -> list[dict]:
    """Localiza a lista de extracoes, aceitando alguns nomes alternativos.

    O prompt pede `extracoes`, mas um modelo em ingles devolve `extractions`
    com alguma frequencia. Aceitar o sinonimo custa uma linha e evita descartar
    uma resposta que esta correta no conteudo.
    """
    for chave in ("extracoes", "extractions", "extraction", "results", "data"):
        valor = dados.get(chave)
        if isinstance(valor, list):
            return [x for x in valor if isinstance(x, dict)]
    return []


# ---------------------------------------------------------------------
# 2. Validacao pelo schema
# ---------------------------------------------------------------------


@dataclass
class Rejeitada:
    """Uma extracao que nao entrou no registro, com o motivo."""

    bruto: dict
    motivo: str
    etapa: str  # "schema" ou "ancoragem"


def _derivar_entity_type(bruto: dict, lex: Lexico) -> dict:
    """Preenche `entity_type` quando o modelo o omite ou inventa um valor.

    A derivacao e a mesma da anotacao: o lexico classifica `target_entity`.
    Manter a regra identica dos dois lados evita que o modelo seja penalizado
    por um campo que nem o anotador preencheu a mao.
    """
    bruto = dict(bruto)
    valido = {"element", "element_group", "oxide", "mineral", "other"}
    if bruto.get("entity_type") not in valido:
        bruto["entity_type"] = lex.tipo_de(str(bruto.get("target_entity") or ""))
    return bruto


def validar(
    dados: dict, doc_id: str, lex: Lexico
) -> tuple[Registro, list[Rejeitada]]:
    """Converte o JSON cru em `Registro`, separando o que nao passa no schema."""
    aceitas: list[Extracao] = []
    rejeitadas: list[Rejeitada] = []

    for bruto in extrair_lista(dados):
        try:
            aceitas.append(Extracao(**_derivar_entity_type(bruto, lex)))
        except ValidationError as erro:
            motivos = "; ".join(
                f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in erro.errors()
            )
            rejeitadas.append(Rejeitada(bruto, motivos, "schema"))
        except TypeError as erro:
            rejeitadas.append(Rejeitada(bruto, str(erro), "schema"))

    return Registro(doc_id=doc_id, extracoes=aceitas), rejeitadas


# ---------------------------------------------------------------------
# 3. Ancoragem
# ---------------------------------------------------------------------


def normalizar(texto: str) -> str:
    return ESPACOS.sub(" ", texto).strip()


# \u00b7 e o ponto medio; \u2212 e o sinal de menos tipografico.
PONTO_MEDIO = re.compile(r"(\d)\u00b7\s?(\d)")
MENOS_TIPOGRAFICO = re.compile(r"\u2212\s*(?=\d)|\u2212")


def normalizar_numeros(texto: str) -> str:
    """Grafias de numero que o OpenAlex herda de algumas revistas.

    O Journal of Petrology usa ponto medio como separador decimal ("60<ponto medio> 9%"
    quer dizer 60.9%), e varios resumos trazem o sinal de menos tipografico
    ("<menos tipografico>3.28"). Sem isso, um valor lido corretamente seria rejeitado como
    alucinacao.
    """
    texto = PONTO_MEDIO.sub(r"\1.\2", texto)
    # "\u2212 3.5" (menos tipografico separado do numero) tambem e -3.5.
    return MENOS_TIPOGRAFICO.sub("-", texto)


def valor_aparece(valor: float | None, texto: str) -> bool:
    """O numero aparece no texto, em alguma grafia plausivel?

    Um mesmo valor e escrito de varias formas: 32000 pode estar como "32,000";
    63.0 costuma estar como "63". Sem cobrir essas formas, a verificacao
    rejeitaria extracoes corretas.
    """
    if valor is None:
        return True
    texto = normalizar_numeros(texto)
    formas = {f"{valor:g}", str(valor)}
    if valor == int(valor):
        inteiro = int(valor)
        formas.add(str(inteiro))
        formas.add(f"{inteiro:,}")  # separador de milhar
    return any(forma in texto for forma in formas)


def verificar_ancoragem(
    registro: Registro, resumo: str
) -> tuple[Registro, list[Rejeitada]]:
    """Mantem apenas extracoes verificaveis contra o texto de origem."""
    alvo = normalizar(resumo)
    aceitas: list[Extracao] = []
    rejeitadas: list[Rejeitada] = []

    for extracao in registro.extracoes:
        sentenca = normalizar(extracao.sentence)
        if sentenca not in alvo:
            rejeitadas.append(
                Rejeitada(
                    json.loads(extracao.model_dump_json()),
                    "`sentence` nao e trecho literal do resumo",
                    "ancoragem",
                )
            )
            continue
        if not valor_aparece(extracao.value, sentenca):
            rejeitadas.append(
                Rejeitada(
                    json.loads(extracao.model_dump_json()),
                    f"`value` {extracao.value} nao aparece na sentenca citada",
                    "ancoragem",
                )
            )
            continue
        aceitas.append(extracao)

    return Registro(doc_id=registro.doc_id, extracoes=aceitas), rejeitadas


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


@dataclass
class Resultado:
    """O que saiu de um documento, com o rastro de tudo que foi descartado.

    `erro` preenchido significa que o modelo nao entregou resposta utilizavel,
    o que e diferente de entregar uma lista vazia. Os dois casos precisam ser
    contados separadamente: um e falha de infraestrutura, o outro e o
    julgamento do modelo.
    """

    doc_id: str
    registro: Registro
    rejeitadas: list[Rejeitada] = field(default_factory=list)
    tentativas: int = 1
    erro: str | None = None
    respostas: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    max_tokens: int | None = None

    @property
    def n(self) -> int:
        return self.registro.n

    @property
    def falhou(self) -> bool:
        return self.erro is not None

    def para_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "extracoes": json.loads(self.registro.model_dump_json())["extracoes"],
            "rejeitadas": [
                {"bruto": r.bruto, "motivo": r.motivo, "etapa": r.etapa}
                for r in self.rejeitadas
            ],
            "tentativas": self.tentativas,
            "erro": self.erro,
            "finish_reason": self.finish_reason,
            "max_tokens": self.max_tokens,
        }


def carregar_prompt(nome: str) -> str:
    """Le um prompt versionado de `prompts/`. Aceita nome com ou sem extensao."""
    caminho = PROMPTS / (nome if nome.endswith(".txt") else f"{nome}.txt")
    return caminho.read_text(encoding="utf-8").strip()


def teto_max_tokens() -> int:
    return int(config_llm().get("max_tokens_teto") or TETO_MAX_TOKENS_PADRAO)


def extrair(
    resumo: str,
    doc_id: str,
    prompt: str,
    *,
    lex: Lexico | None = None,
    tentativas: int = TENTATIVAS_PADRAO,
    verificar: bool = True,
    max_tokens: int | None = None,
    semente: int = 0,
    **kw,
) -> Resultado:
    """Extrai de um resumo, com re-tentativa adequada a cada tipo de falha.

    - resposta cortada pelo limite de tokens: repete a mesma pergunta com o
      dobro do orcamento, ate o teto da config;
    - resposta mal formatada: devolve ao modelo a mensagem de erro concreta.

    Se nem assim vier resposta valida, o documento e registrado com `erro` e
    o lote segue.

    `semente` separa repeticoes da mesma avaliacao: com temperatura minima de
    0,5 cada semente e uma amostra nova, com cache proprio. A semente 0 usa as
    mesmas chaves de cache de antes.
    """
    lex = lex or Lexico.carregar()
    orcamento = max_tokens or max_tokens_padrao()
    teto = max(teto_max_tokens(), orcamento)
    mensagens = [{"role": "user", "content": resumo}]
    respostas: list[str] = []
    fim: str | None = None

    for tentativa in range(1, tentativas + 1):
        registro_api = perguntar_completo(
            mensagens, sistema=prompt, run=semente * 100 + tentativa - 1,
            max_tokens=orcamento, **kw
        )
        resposta, fim = registro_api["texto"], registro_api.get("finish_reason")
        respostas.append(resposta or "")

        try:
            dados = ler_json(resposta)
        except RespostaInvalida as erro:
            truncada = fim == "length"
            motivo = f"truncada em max_tokens={orcamento}" if truncada else str(erro)
            sem_saida = truncada and orcamento >= teto
            if tentativa == tentativas or sem_saida:
                return Resultado(
                    doc_id, Registro(doc_id=doc_id), tentativas=tentativa, erro=motivo,
                    respostas=respostas, finish_reason=fim, max_tokens=orcamento,
                )
            if truncada:
                orcamento = min(orcamento * 2, teto)
            else:
                mensagens = [
                    *mensagens,
                    {"role": "assistant", "content": resposta or ""},
                    {"role": "user", "content":
                     f"A resposta nao pode ser lida: {erro}. "
                     "Responda apenas com o JSON pedido, sem texto em volta."},
                ]
            continue

        registro, rejeitadas = validar(dados, doc_id, lex)
        if verificar:
            registro, rejeitadas_ancora = verificar_ancoragem(registro, resumo)
            rejeitadas = [*rejeitadas, *rejeitadas_ancora]
        return Resultado(doc_id, registro, rejeitadas, tentativa, None, respostas,
                         finish_reason=fim, max_tokens=orcamento)

    return Resultado(doc_id, Registro(doc_id=doc_id), tentativas=tentativas,
                     erro="sem resposta", respostas=respostas,
                     finish_reason=fim, max_tokens=orcamento)


def resumo_de_rejeicoes(resultados: list[Resultado]) -> dict[str, int]:
    """Contagem por etapa de rejeicao. Alimenta a analise de erro da F3-06."""
    contagem: dict[str, int] = {}
    for resultado in resultados:
        for rejeitada in resultado.rejeitadas:
            contagem[rejeitada.etapa] = contagem.get(rejeitada.etapa, 0) + 1
    return contagem


def carregar_resultados_llm(caminho: Path | str) -> dict[str, Registro]:
    """Le um JSONL de saida do lote e devolve `Registro` por documento.

    Formato aceito: a linha do `processar_lote` (com `id` e `resultado`) ou o
    proprio dicionario de `Resultado.para_json`.
    """
    registros: dict[str, Registro] = {}
    caminho = Path(caminho)
    if not caminho.exists():
        return registros
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        try:
            bruto = json.loads(linha)
        except json.JSONDecodeError:
            continue
        dados = bruto.get("resultado") if "resultado" in bruto else bruto
        if not isinstance(dados, dict) or "doc_id" not in dados:
            continue
        registros[dados["doc_id"]] = Registro(
            doc_id=dados["doc_id"], extracoes=dados.get("extracoes", [])
        )
    return registros
