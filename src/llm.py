"""Cliente da IlumA, cache e execucao em lote (F(-1)-03 a F(-1)-06).

A IlumA expoe um proxy LiteLLM compativel com a API da OpenAI, entao o
cliente oficial `openai` funciona apontando `base_url` para o endpoint do
CNPEM. O unico segredo necessario e o token pessoal, lido do `.env`.

Uso tipico:

    from src.llm import perguntar, processar_lote, estimar_custo

    estimar_custo(resumos, sistema=PROMPT)      # antes de gastar cota
    perguntar("Responda apenas: conexao OK.")   # chamada avulsa

Verificacao da conexao pela linha de comando:

    python -m src.llm
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

from openai import APIStatusError, AuthenticationError, OpenAI
from tqdm import tqdm

from src.config import RAIZ, carregar_config, obter_env

# Valores de fallback, usados apenas se o .env nao trouxer os campos.
BASE_URL_PADRAO = "https://iluma.cnpem.br:4000/v1"
MODELO_PADRAO = "iluma"

# Esta instalacao rejeita temperatura abaixo deste valor. A consequencia e que
# nenhuma medicao e deterministica, e por isso toda metrica do projeto precisa
# de repeticoes (ver F4-07 e F5-08).
TEMPERATURA_PISO = 0.5

# Orcamento de saida quando a config nao define um. Ver `max_tokens_padrao`.
MAX_TOKENS_PADRAO = 2000

DIR_CACHE = RAIZ / "cache"

# Heuristica de tokenizacao para estimativa previa. Textos cientificos em
# ingles ficam perto de 4 caracteres por token. Serve para dimensionar, nao
# para faturar.
CARACTERES_POR_TOKEN = 4


class TokenAusente(RuntimeError):
    """Levantada quando o ILUMA_TOKEN nao foi configurado."""


_INSTRUCOES_TOKEN = f"""
ILUMA_TOKEN nao encontrado.

Para configurar:

  1. Acesse https://iluma.cnpem.br e faca login com a conta do CNPEM.
  2. No Console, va em "Virtual Keys" e clique em "Create New Key".
  3. Copie a chave gerada (comeca com "sk-"). Ela so aparece uma vez.
  4. Na raiz do repositorio:

         cp .env.example .env

  5. Abra o .env e cole a chave em ILUMA_TOKEN.

O arquivo esperado e: {RAIZ / ".env"}
O .env ja esta no .gitignore e nao vai para o repositorio.
""".strip()


# =====================================================================
# Configuracao
# =====================================================================


def config_llm() -> dict:
    """Secao `llm` do config.yaml, ou vazio se a config nao existir."""
    try:
        return carregar_config().get("llm", {}) or {}
    except Exception:  # noqa: BLE001 - config ausente ou malformada cai nos padroes
        return {}


def obter_token() -> str:
    """Devolve o token da IlumA ou explica como obte-lo."""
    token = obter_env("ILUMA_TOKEN")
    if not token or token.startswith("sk-cole-seu-token"):
        raise TokenAusente(_INSTRUCOES_TOKEN)
    return token


def obter_modelo() -> str:
    return obter_env("ILUMA_MODELO") or MODELO_PADRAO


def max_tokens_padrao() -> int:
    return int(config_llm().get("max_tokens") or MAX_TOKENS_PADRAO)


def parametros_extras_padrao() -> dict:
    """Parametros repassados ao servidor sem interpretacao, via `extra_body`.

    Existe para opcoes que dependem do modelo servido, como
    `reasoning_effort` em modelos que raciocinam antes de responder. Ficam na
    config, e nao no codigo, porque mudam o comportamento do modelo e precisam
    ser congeladas junto com o resto da configuracao (F4-12).
    """
    return dict(config_llm().get("parametros_extras") or {})


def ajustar_temperatura(temperatura: float | None) -> float:
    """Aplica o piso da instalacao, avisando quando o valor pedido e menor."""
    if temperatura is None:
        temperatura = float(config_llm().get("temperatura") or TEMPERATURA_PISO)
    if temperatura < TEMPERATURA_PISO:
        warnings.warn(
            f"temperatura {temperatura} abaixo do piso da IlumA; usando {TEMPERATURA_PISO}",
            stacklevel=2,
        )
        return TEMPERATURA_PISO
    return temperatura


def criar_cliente() -> OpenAI:
    """Cliente OpenAI apontado para o proxy da IlumA."""
    return OpenAI(
        api_key=obter_token(),
        base_url=obter_env("ILUMA_BASE_URL") or BASE_URL_PADRAO,
    )


# =====================================================================
# F(-1)-04 - Cache de chamadas
# =====================================================================


def chave_cache(
    mensagens: list[dict],
    modelo: str,
    temperatura: float,
    max_tokens: int,
    run: int,
    extras: dict | None = None,
) -> str:
    """SHA-256 do que define univocamente uma chamada.

    O campo `run` e essencial. Com temperatura 0.5 uma mesma pergunta produz
    respostas diferentes, e a autoconsistencia da F4-07 depende de guardar
    cada amostra separadamente. Sem `run` na chave, o cache devolveria sempre
    a primeira amostra e destruiria a medicao de variancia da F5-08.

    Os `extras` so entram na chave quando existem, para que o cache gravado
    antes deles continue valido.
    """
    conteudo = {
        "modelo": modelo,
        "temperatura": temperatura,
        "max_tokens": max_tokens,
        "mensagens": mensagens,
        "run": run,
    }
    if extras:
        conteudo["extras"] = extras
    payload = json.dumps(conteudo, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _caminho_cache(chave: str, dir_cache: Path | None = None) -> Path:
    return (dir_cache or DIR_CACHE) / f"{chave}.json"


def ler_cache(chave: str, dir_cache: Path | None = None) -> dict | None:
    caminho = _caminho_cache(chave, dir_cache)
    if not caminho.exists():
        return None
    try:
        registro = json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Arquivo truncado por interrupcao. Trata como ausente.
        return None
    if not (registro.get("texto") or "").strip():
        # Resposta vazia gravada antes desta correcao: servi-la de novo so
        # congelaria a falha. Tratada como ausente, a chamada e refeita.
        return None
    return registro


def gravar_cache(chave: str, registro: dict, dir_cache: Path | None = None) -> None:
    destino = dir_cache or DIR_CACHE
    destino.mkdir(parents=True, exist_ok=True)
    caminho = _caminho_cache(chave, destino)
    # Grava em temporario e renomeia: evita cache corrompido se o processo morrer.
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporario.replace(caminho)


def limpar_cache(dir_cache: Path | None = None) -> int:
    """Apaga o cache inteiro. Devolve quantos arquivos removeu."""
    destino = dir_cache or DIR_CACHE
    arquivos = list(destino.glob("*.json"))
    for arquivo in arquivos:
        arquivo.unlink()
    return len(arquivos)


def resposta_utilizavel(registro: dict) -> bool:
    """A resposta vale a pena ser guardada?

    Nao vale quando veio vazia ou foi cortada pelo limite de tokens. Guardar
    uma resposta assim no cache faria toda reexecucao com os mesmos
    parametros devolver a mesma falha, sem nem chamar o modelo.
    """
    return bool((registro.get("texto") or "").strip()) and registro.get(
        "finish_reason"
    ) != "length"


# =====================================================================
# Chamada ao modelo
# =====================================================================

# Contador da sessao. Alimenta o rastreio de orcamento da F5-03. A trava
# existe porque o lote pode rodar em varias threads.
USO_SESSAO: dict[str, int] = {
    "chamadas_api": 0,
    "acertos_cache": 0,
    "tokens_entrada": 0,
    "tokens_saida": 0,
}
_TRAVA_USO = threading.Lock()


def _contar(**incrementos: int) -> None:
    with _TRAVA_USO:
        for chave, valor in incrementos.items():
            USO_SESSAO[chave] += valor


def zerar_uso() -> None:
    with _TRAVA_USO:
        for chave in USO_SESSAO:
            USO_SESSAO[chave] = 0


def _raciocinio_de(mensagem: Any) -> str:
    """Texto de raciocinio, quando o servidor o expoe separado da resposta.

    Modelos que raciocinam antes de responder costumam devolver esse texto em
    um campo proprio (`reasoning_content` ou `reasoning`), fora de `content`.
    """
    for nome in ("reasoning_content", "reasoning"):
        valor = getattr(mensagem, nome, None)
        if valor is None:
            valor = (getattr(mensagem, "model_extra", None) or {}).get(nome)
        if isinstance(valor, str) and valor:
            return valor
    return ""


def _chamar_api(
    mensagens: list[dict],
    modelo: str,
    temperatura: float,
    max_tokens: int,
    extras: dict | None = None,
) -> dict:
    """Chamada crua, sem cache. E o unico ponto que toca a rede.

    Os testes substituem esta funcao para exercitar cache e lote offline.

    Devolve tambem `finish_reason`: "length" significa que a resposta foi
    cortada pelo limite de tokens, e e o sinal que distingue truncamento de
    resposta simplesmente mal formatada.
    """
    argumentos = {
        "model": modelo,
        "messages": mensagens,
        "temperature": temperatura,
        "max_tokens": max_tokens,
    }
    if extras:
        argumentos["extra_body"] = extras
    resposta = criar_cliente().chat.completions.create(**argumentos)

    escolha = resposta.choices[0]
    uso = getattr(resposta, "usage", None)
    detalhes = getattr(uso, "completion_tokens_details", None)
    return {
        "texto": escolha.message.content,
        "finish_reason": getattr(escolha, "finish_reason", None),
        "tokens_entrada": getattr(uso, "prompt_tokens", 0) or 0,
        "tokens_saida": getattr(uso, "completion_tokens", 0) or 0,
        "tokens_raciocinio": getattr(detalhes, "reasoning_tokens", 0) or 0,
        "caracteres_raciocinio": len(_raciocinio_de(escolha.message)),
    }


def perguntar_completo(
    mensagens: str | list[dict],
    *,
    sistema: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    modelo: str | None = None,
    run: int = 0,
    usar_cache: bool = True,
    dir_cache: Path | None = None,
    parametros_extras: dict | None = None,
) -> dict:
    """Igual a `perguntar`, mas devolve o registro completo.

    Chaves do retorno: texto, finish_reason, tokens_entrada, tokens_saida,
    tokens_raciocinio, do_cache, chave.

    `max_tokens` e `parametros_extras`, quando omitidos, vem da config.
    """
    if isinstance(mensagens, str):
        mensagens = [{"role": "user", "content": mensagens}]
    if sistema:
        mensagens = [{"role": "system", "content": sistema}, *mensagens]

    modelo = modelo or obter_modelo()
    temperatura = ajustar_temperatura(temperatura)
    max_tokens = max_tokens or max_tokens_padrao()
    extras = parametros_extras if parametros_extras is not None else parametros_extras_padrao()
    chave = chave_cache(mensagens, modelo, temperatura, max_tokens, run, extras)

    if usar_cache:
        guardado = ler_cache(chave, dir_cache)
        if guardado is not None:
            _contar(acertos_cache=1)
            return {**guardado, "do_cache": True, "chave": chave}

    # Os extras so sao passados quando existem: mantem compativeis os
    # transportes falsos dos testes, que recebem quatro argumentos.
    if extras:
        resultado = _chamar_api(mensagens, modelo, temperatura, max_tokens, extras)
    else:
        resultado = _chamar_api(mensagens, modelo, temperatura, max_tokens)
    _contar(
        chamadas_api=1,
        tokens_entrada=resultado.get("tokens_entrada", 0),
        tokens_saida=resultado.get("tokens_saida", 0),
    )

    registro = {
        **resultado,
        "modelo": modelo,
        "temperatura": temperatura,
        "max_tokens": max_tokens,
        "run": run,
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if usar_cache and resposta_utilizavel(registro):
        gravar_cache(chave, registro, dir_cache)
    return {**registro, "do_cache": False, "chave": chave}


def perguntar(mensagens: str | list[dict], **kwargs) -> str:
    """Envia mensagens ao modelo e devolve apenas o texto da resposta."""
    return perguntar_completo(mensagens, **kwargs)["texto"]


# =====================================================================
# F(-1)-05 - Lote com retomada
# =====================================================================


def _sanear_jsonl(caminho: Path) -> None:
    """Remove uma ultima linha incompleta deixada por interrupcao.

    Sem isso o proximo append gruda no fragmento e produz uma linha
    irrecuperavel, perdendo silenciosamente o registro seguinte.
    """
    if not caminho.exists() or caminho.stat().st_size == 0:
        return
    with caminho.open("rb+") as arquivo:
        arquivo.seek(-1, 2)
        if arquivo.read(1) == b"\n":
            return
        conteudo = caminho.read_bytes()
        corte = conteudo.rfind(b"\n")
        arquivo.truncate(corte + 1)


def ids_processados(saida: str | Path) -> set[str]:
    """Le o JSONL de saida e devolve os ids ja concluidos."""
    caminho = Path(saida)
    if not caminho.exists():
        return set()
    feitos = set()
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            feitos.add(json.loads(linha)["id"])
        except (json.JSONDecodeError, KeyError):
            # Ultima linha truncada por interrupcao: sera refeita.
            continue
    return feitos


def carregar_resultados(saida: str | Path) -> list[dict]:
    """Le o JSONL de saida inteiro, ignorando linhas truncadas."""
    caminho = Path(saida)
    if not caminho.exists():
        return []
    registros = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha:
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError:
                continue
    return registros


def processar_lote(
    itens: Iterable[Any],
    processar: Callable[[Any], Any],
    saida: str | Path,
    *,
    id_de: Callable[[Any], str] = lambda item: str(item["doc_id"]),
    refazer: bool = False,
    pausa_s: float = 0.0,
    descricao: str = "processando",
    concorrencia: int = 1,
) -> list[dict]:
    """Aplica `processar` a cada item, gravando linha a linha com retomada.

    Cada linha do JSONL tem: id, resultado, erro, instante. O arquivo e aberto
    em modo append e sofre flush a cada item, entao matar o processo no meio
    perde no maximo os itens em andamento. Rodar de novo pula o que ja foi
    feito.

    Com `concorrencia` > 1, varias chamadas correm em paralelo. A gravacao
    continua acontecendo so na thread principal, na ordem em que os itens
    terminam, entao o arquivo nunca recebe duas escritas ao mesmo tempo.

    Falhas individuais nao interrompem o lote: ficam registradas no campo
    `erro`.
    """
    itens = list(itens)
    caminho = Path(saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    if not refazer:
        _sanear_jsonl(caminho)
    feitos = set() if refazer else ids_processados(caminho)
    pendentes = [item for item in itens if id_de(item) not in feitos]

    if feitos:
        print(f"{len(feitos)} ja processados, {len(pendentes)} pendentes")

    def executar(item: Any) -> tuple[Any, str | None]:
        try:
            return processar(item), None
        except Exception as excecao:  # noqa: BLE001 - falha isolada nao derruba o lote
            return None, f"{type(excecao).__name__}: {excecao}"

    modo = "w" if refazer else "a"
    with caminho.open(modo, encoding="utf-8") as arquivo:

        def gravar(item: Any, resultado: Any, erro: str | None) -> None:
            arquivo.write(
                json.dumps(
                    {
                        "id": id_de(item),
                        "resultado": resultado,
                        "erro": erro,
                        "instante": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            arquivo.flush()

        if concorrencia <= 1:
            for item in tqdm(pendentes, desc=descricao):
                resultado, erro = executar(item)
                gravar(item, resultado, erro)
                if pausa_s:
                    time.sleep(pausa_s)
        else:
            executor = ThreadPoolExecutor(max_workers=concorrencia)
            try:
                futuros = {executor.submit(executar, item): item for item in pendentes}
                for futuro in tqdm(as_completed(futuros), total=len(futuros), desc=descricao):
                    resultado, erro = futuro.result()
                    gravar(futuros[futuro], resultado, erro)
            finally:
                # Numa interrupcao, nao espera as chamadas que ainda estao na fila.
                executor.shutdown(wait=False, cancel_futures=True)

    return carregar_resultados(caminho)


# =====================================================================
# F(-1)-06 - Estimativa de custo
# =====================================================================


def estimar_tokens(texto: str) -> int:
    """Estimativa grosseira por contagem de caracteres.

    Arredonda para cima, mas texto vazio custa zero -- sem isso, um prompt
    de sistema ausente somaria um token fantasma a cada chamada.
    """
    if not texto:
        return 0
    return math.ceil(len(texto) / CARACTERES_POR_TOKEN)


def estimar_custo(
    textos: Iterable[str],
    *,
    sistema: str = "",
    exemplos: str = "",
    max_tokens_saida: int = 2000,
    k: int = 1,
    verboso: bool = True,
) -> dict:
    """Dimensiona um lote antes de dispara-lo.

    `k` e o numero de amostras por item (autoconsistencia da F4-07).
    Os tokens de saida sao um teto, nao uma previsao: quase sempre o modelo
    responde bem menos que `max_tokens`.
    """
    textos = list(textos)
    fixo = estimar_tokens(sistema) + estimar_tokens(exemplos)
    entrada = sum(estimar_tokens(t) + fixo for t in textos) * k
    chamadas = len(textos) * k
    resumo = {
        "itens": len(textos),
        "k": k,
        "chamadas": chamadas,
        "tokens_entrada": entrada,
        "tokens_saida_teto": chamadas * max_tokens_saida,
        "tokens_fixos_por_chamada": fixo,
    }
    if verboso:
        print(f"itens               : {resumo['itens']}")
        print(f"amostras por item   : {k}")
        print(f"chamadas            : {chamadas:,}")
        print(f"tokens de entrada   : {entrada:,} (estimado)")
        print(f"tokens de saida     : ate {resumo['tokens_saida_teto']:,}")
        print("Confira o consumo real em Console -> Usage Logs depois do piloto.")
    return resumo


def resumo_uso(verboso: bool = True) -> dict:
    """Consumo acumulado desde o inicio da sessao (ou desde `zerar_uso`)."""
    if verboso:
        print(f"chamadas a API   : {USO_SESSAO['chamadas_api']:,}")
        print(f"acertos de cache : {USO_SESSAO['acertos_cache']:,}")
        print(f"tokens entrada   : {USO_SESSAO['tokens_entrada']:,}")
        print(f"tokens saida     : {USO_SESSAO['tokens_saida']:,}")
    return dict(USO_SESSAO)


# =====================================================================
# F(-1)-03 - Verificacao da conexao
# =====================================================================


def testar_conexao(verboso: bool = True) -> bool:
    """Criterio de conclusao da F(-1)-03.

    Devolve True se o modelo respondeu a frase combinada.
    """
    esperado = "conexao OK"

    def log(msg: str) -> None:
        if verboso:
            print(msg)

    try:
        token = obter_token()
    except TokenAusente as erro:
        log(str(erro))
        return False

    log(f"token      : {token[:7]}...{token[-4:]} ({len(token)} caracteres)")
    log(f"base_url   : {obter_env('ILUMA_BASE_URL') or BASE_URL_PADRAO}")
    log(f"modelo     : {obter_modelo()}")
    log(f"temperatura: {ajustar_temperatura(None)}")
    log("")

    try:
        resposta = perguntar(
            f"Responda exatamente com esta frase, sem nada mais: {esperado}.",
            max_tokens=50,
            usar_cache=False,
        )
    except AuthenticationError:
        log("FALHA: token recusado pela IlumA. Gere uma chave nova no Console.")
        return False
    except APIStatusError as erro:
        log(f"FALHA: a IlumA respondeu HTTP {erro.status_code}.")
        log(f"       {erro.message}")
        return False
    except Exception as erro:  # noqa: BLE001 - diagnostico: rede, DNS, VPN, certificado
        log(f"FALHA: nao foi possivel falar com a IlumA ({type(erro).__name__}).")
        log(f"       {erro}")
        log("       Verifique a conexao e se o acesso exige a rede do CNPEM.")
        return False

    log(f"resposta   : {resposta!r}")
    ok = esperado.lower() in (resposta or "").lower()
    log("")
    log("CONEXAO OK" if ok else "Conectou, mas a resposta veio diferente do esperado.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if testar_conexao() else 1)
