"""
Classe para baixar metadados e abstracts de artigos da OpenAlex
(https://openalex.org), a partir de uma ou mais palavras-chave.

IMPORTANTE (atualizado - fev/2026):
- Desde 13/fev/2026, a OpenAlex EXIGE uma API key para todas as requisições.
- A key é gratuita: crie uma conta em openalex.org e pegue a sua em
  https://openalex.org/settings/api
- Você recebe $1 de uso grátis por dia. Buscas (search=) custam $1 a cada
  1000 chamadas; listagem com filtro custa $0.10 a cada 1000 chamadas.
- O dado bruto continua livre via snapshot completo (download em bulk),
  caso você precise de volumes muito grandes sem se preocupar com custo.

NOTA SOBRE "AUTHOR KEYWORDS":
A OpenAlex não distribui palavras-chave escolhidas pelo autor no momento
da submissão (o que normalmente vem do Scopus/Web of Science/editor).
O que ela oferece é o campo `keywords`, uma lista de termos extraídos
automaticamente por um algoritmo da própria OpenAlex a partir do título
e do abstract, cada um com um score de confiança. O script inclui esse
campo como `palavras_chave_openalex` — trate-o como palavras-chave
algorítmicas, não como author keywords no sentido estrito.

NOTA SOBRE ABSTRACTS AUSENTES:
Nem todo artigo indexado pela OpenAlex tem abstract disponível. Os motivos
mais comuns são restrição de copyright do editor (a OpenAlex não pode
redistribuir o texto) e ausência do abstract na fonte original agregada
(Crossref, PubMed, etc.). Isso não é um erro do script — é uma
característica dos dados. O script reporta ao final quantos artigos
vieram sem abstract, para que você decida se descarta essas linhas antes
de alimentar o pipeline de extração.

NOTA SOBRE "ABSTRACTS" SUJOS:
Alguns editores submetem para o Crossref/PubMed, no campo de abstract,
lixo de página (menus de navegação, texto de paywall como "you do not
have access", botões de "download citation") em vez do resumo de fato,
ou um preview truncado terminando em "...". O script filtra esses casos
automaticamente (heurística por padrões conhecidos + reticências no
final) e trata como se o artigo não tivesse abstract — ou seja, esses
artigos também contam na estatística de "sem abstract" e são descartados
quando require_abstract=True.

NOTA SOBRE BUSCA BOOLEANA:
Por padrão (`keywords`), os termos são combinados só com OR — o que pode
trazer falsos positivos quando uma keyword é uma sigla curta e ambígua
(ex.: "REE" também aparece em "resting energy expenditure", nada a ver
com "rare earth elements"). Para exigir que a sigla apareça junto de um
termo de contexto, use `query` (biblioteca) ou -Q (CLI) com a sintaxe
booleana nativa da OpenAlex: AND / OR / NOT em MAIÚSCULAS, parênteses
para agrupar, aspas duplas para frase exata. Ex.:

    ("REE" OR "REO" OR "TREO") AND ("rare earth" OR mining OR ore OR deposit)

`keywords` e `query` são mutuamente exclusivos — informe só um dos dois.

Uso como biblioteca:
    from openalex_downloader import OpenAlexDownloader

    # Busca simples (OR entre keywords)
    downloader = OpenAlexDownloader(
        api_key="SUA_API_KEY",
        keywords=["soft robotics", "soft actuators"],
        email="seuemail@exemplo.com",   # opcional, recomendado
        max_results=500,                # opcional, None = todos os resultados
        open_access_only=True,          # opcional, só Open Access
        require_abstract=True,          # opcional, descarta sem abstract e
                                         # continua até juntar 500 COM abstract
    )
    artigos = downloader.buscar()
    downloader.salvar_csv(artigos, "resultado.csv")

    # Busca booleana (exige contexto junto da sigla)
    downloader = OpenAlexDownloader(
        api_key="SUA_API_KEY",
        query='("REE" OR "REO" OR "TREO") AND ("rare earth" OR mining OR ore)',
    )

Uso via terminal (CLI):
    python openalex_downloader.py -k SUA_API_KEY -w "soft robotics" "soft actuators"

    Argumentos:
      -k, --api-key       API key da OpenAlex (obrigatório)
      -w, --keywords      Uma ou mais palavras-chave, combinadas com OR
                           (obrigatório, a menos que use -Q)
      -Q, --query         String de busca booleana pronta, sintaxe da
                           própria OpenAlex (AND/OR/NOT maiúsculos,
                           parênteses, aspas). Alternativa a -w.
      -e, --email         E-mail para o polite pool (opcional)
      -m, --max-results   Baixa apenas os N resultados mais relevantes
                           (opcional; se omitido, baixa todos os resultados
                           encontrados, sem ordenação específica)
      -p, --per-page      Resultados por página, máx. 200 (opcional)
      -o, --output        Caminho do CSV de saída (opcional)
      -q, --quiet         Não imprime progresso artigo a artigo
      --open-access       Retorna apenas artigos em acesso aberto
      --require-abstract  Descarta artigos sem abstract e continua
                           buscando até juntar -m artigos COM abstract
                           (em vez de -m artigos no total)

    Exemplo completo:
      python openalex_downloader.py -k MH6pqjJvJJTPPU0W87AkYz \
          -w "soft robotics" -e seuemail@exemplo.com \
          --open-access --require-abstract -m 500 -o soft_robotics.csv

    Ver todas as opções:
      python openalex_downloader.py --help

    openalex_downloader.py -k MH6pqjJvJJTPPU0W87AkYz --% -Q "(\"REE\" OR \"REO\" OR \"TREO\") AND (\"rare earth\" OR \"Total Rare Earth Oxides\")" --require-abstract -m 1000 -o TERRAS_RARAS.csv --open-access    
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class OpenAlexDownloader:
    """
    Encapsula a busca de artigos (works) na API da OpenAlex e a
    reconstrução/exportação dos abstracts.

    A ordenação é sempre por relevância à busca (`relevance_score:desc`),
    o que garante que `max_results` sempre traga os N artigos mais
    relevantes para as palavras-chave informadas, e não uma amostra
    arbitrária.
    """

    BASE_URL = "https://api.openalex.org/works"
    SORT = "relevance_score:desc"

    def __init__(
        self,
        api_key: str,
        keywords: str | list[str] | None = None,
        query: str | None = None,
        email: str | None = None,
        max_results: int | None = None,
        per_page: int = 200,
        open_access_only: bool = False,
        require_abstract: bool = False,
    ) -> None:
        """
        Parâmetros
        ----------
        api_key : str
            Sua API key da OpenAlex (obrigatória desde fev/2026).
            Gere em https://openalex.org/settings/api
        keywords : str ou list[str], opcional
            Uma palavra-chave (str) ou lista de palavras-chave. Quando é
            uma lista, os termos são combinados com "OR" na busca, ou seja,
            traz artigos que contenham qualquer um dos termos. Use isso
            para buscas simples. Ignorado se `query` for informado.
        query : str, opcional
            Uma string de busca booleana JÁ MONTADA por você, usando a
            sintaxe da própria OpenAlex: operadores AND / OR / NOT (em
            MAIÚSCULAS), parênteses para agrupar, e aspas duplas para
            frase exata. Use isso quando o "OR simples" de `keywords` não
            for suficiente — por exemplo, para exigir que uma sigla curta
            (REE, REO) apareça junto de um termo de contexto, evitando
            falsos positivos de outras áreas:

                query='("REE" OR "REO" OR "TREO" OR "Total Rare Earth '
                      'Oxides") AND ("rare earth" OR mining OR ore OR '
                      'deposit OR grade)'

            Quando `query` é informado, `keywords` é ignorado — a string
            é enviada como está para o parâmetro `search` da API.
        email : str, opcional
            Seu e-mail, usado no "polite pool" da OpenAlex (não é
            obrigatório mas é recomendado).
        max_results : int, opcional
            Número máximo de artigos a baixar, sempre os mais relevantes
            para as keywords. None = baixa todos os resultados encontrados.
        per_page : int
            Resultados por página (máximo permitido pela API: 200).
        open_access_only : bool, opcional
            Se True, retorna apenas artigos em acesso aberto (Open Access).
        require_abstract : bool, opcional
            Se True, artigos sem abstract são descartados e a busca
            continua paginando até juntar `max_results` artigos COM
            abstract (ou até a busca se esgotar). Ou seja, `max_results`
            passa a significar "quantos abstracts eu quero", não "quantos
            artigos baixar". Sem efeito se `max_results` for None.
        """
        if not api_key or api_key == "COLOQUE_SUA_API_KEY_AQUI":
            raise ValueError(
                "É necessário informar uma API key válida da OpenAlex. "
                "Gere a sua gratuitamente em https://openalex.org/settings/api"
            )

        if query is not None:
            if not query.strip():
                raise ValueError("query não pode ser uma string vazia.")
            keywords = None
        else:
            if isinstance(keywords, str):
                keywords = [keywords]
            if not keywords:
                raise ValueError("Informe `keywords` ou `query`.")

        if max_results is not None and max_results <= 0:
            raise ValueError("max_results deve ser um número positivo.")

        if not 1 <= per_page <= 200:
            raise ValueError("per_page deve estar entre 1 e 200.")

        self.api_key = api_key
        self.keywords = keywords
        self.query = query
        self.email = email
        self.max_results = max_results
        self.per_page = per_page
        self.open_access_only = open_access_only
        self.require_abstract = require_abstract

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _build_query_string(self) -> str:
        """
        Monta a string de busca. Se `query` foi informado no construtor,
        ele é usado literalmente (você já escreveu a sintaxe booleana da
        OpenAlex). Caso contrário, cai no comportamento simples: múltiplas
        `keywords` são combinadas com OR.
        Ex.: ["soft robotics", "soft actuators"] ->
             '"soft robotics" OR "soft actuators"'
        """
        if self.query is not None:
            return self.query
        termos = [f'"{kw}"' if " " in kw else kw for kw in self.keywords]
        return " OR ".join(termos)

    def _build_filter_string(self) -> str | None:
        """
        Monta a string de filtros (parâmetro `filter=`). Atualmente o
        único filtro suportado é acesso aberto. Retorna None se o filtro
        não estiver ativo.
        """
        if self.open_access_only:
            return "open_access.is_oa:true"
        return None

    # Trechos que denunciam "abstract" sujo: dump de página do editor
    # (menus, paywall, citation manager) em vez do resumo de fato. Isso é
    # sujeira que vem de dados mal submetidos pelo próprio editor via
    # Crossref/PubMed — a OpenAlex só repassa o que recebeu.
    _MARCADORES_LIXO = (
        "search for other works by this author",
        "google scholar",
        "download citation file",
        "you do not have access to this content",
        "institutional administrator",
        "add to citation manager",
        "cite view this citation",
        "share icon share",
        "toolbar search",
        "search dropdown menu",
        "advanced search",
        "get permissions",
        "article history first online",
    )

    # Detecta lista de autores no formato "J. Sobrenome;R. Sobrenome;..."
    # colada como se fosse abstract (outro tipo comum de metadado sujo,
    # sem os marcadores de página acima). 4+ ocorrências é um forte sinal
    # de que é uma lista de autores, não um resumo em prosa.
    _PADRAO_LISTA_AUTORES = re.compile(r"(?:[A-Z]\.\s?){1,3}[A-Za-z\-]+;")
    _MIN_OCORRENCIAS_AUTORES = 4

    @classmethod
    def _abstract_e_valido(cls, abstract: str) -> bool:
        """
        Heurística para descartar "abstracts" que na verdade são lixo de
        página (navegação, paywall), uma lista de autores colada, ou um
        preview truncado terminando em reticências — todos comuns em
        metadados mal submetidos por alguns editores. Retorna False
        nesses casos, tratando o artigo como se não tivesse abstract.
        """
        if not abstract:
            return False
        if abstract.rstrip().endswith(("...", "…")):
            return False
        abstract_lower = abstract.lower()
        if any(marcador in abstract_lower for marcador in cls._MARCADORES_LIXO):
            return False
        if (
            len(cls._PADRAO_LISTA_AUTORES.findall(abstract))
            >= cls._MIN_OCORRENCIAS_AUTORES
        ):
            return False
        return True

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
        """
        A OpenAlex não retorna o abstract como texto corrido, e sim como um
        'inverted index': {palavra: [posições em que aparece]}.
        Este método reconstrói o texto original a partir dele.
        """
        if not inverted_index:
            return ""

        posicoes: list[tuple[int, str]] = []
        for palavra, posicoes_lista in inverted_index.items():
            for pos in posicoes_lista:
                posicoes.append((pos, palavra))

        posicoes.sort(key=lambda item: item[0])
        return " ".join(palavra for _, palavra in posicoes)

    @staticmethod
    def _extract_keywords(keywords_raw: list[dict[str, Any]] | None) -> str:
        """
        Extrai as palavras-chave do campo `keywords` da OpenAlex.
        IMPORTANTE: são palavras-chave extraídas algoritmicamente pela
        OpenAlex (a partir de título/abstract), não author keywords
        originais do artigo. Ver nota no topo do arquivo.
        """
        if not keywords_raw:
            return ""
        return "; ".join(
            kw.get("display_name", "") for kw in keywords_raw if kw.get("display_name")
        )

    def _parse_work(self, work: dict[str, Any]) -> dict[str, Any]:
        """Converte um item 'work' retornado pela API em um dicionário simples."""
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))
        if not self._abstract_e_valido(abstract):
            abstract = ""
        palavras_chave = self._extract_keywords(work.get("keywords"))

        autores = ", ".join(
            authorship["author"]["display_name"]
            for authorship in work.get("authorships", [])
            if authorship.get("author")
        )

        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}

        return {
            "id_openalex": work.get("id"),
            "titulo": work.get("title") or work.get("display_name"),
            "ano": work.get("publication_year"),
            "autores": autores,
            "revista": source.get("display_name"),
            "doi": work.get("doi"),
            "citado_por": work.get("cited_by_count"),
            "abstract": abstract,
            "palavras_chave_openalex": palavras_chave,
        }

    def _build_request_params(self) -> dict[str, Any]:
        """Monta o dicionário de parâmetros comuns a todas as requisições."""
        params: dict[str, Any] = {
            "search": self._build_query_string(),
            "per_page": self.per_page,
            "sort": self.SORT,
            "api_key": self.api_key,
        }
        if self.email:
            params["mailto"] = self.email

        filtro = self._build_filter_string()
        if filtro:
            params["filter"] = filtro

        return params

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def buscar(self, verbose: bool = True) -> list[dict[str, Any]]:
        """
        Executa a busca paginada na OpenAlex e retorna uma lista de
        dicionários com os artigos encontrados (respeitando max_results,
        sempre priorizando os mais relevantes para as keywords).

        Se `require_abstract=True`, artigos sem abstract são descartados
        (não entram nem em `resultados` nem contam para `max_results`) e
        a paginação continua até juntar `max_results` artigos com
        abstract, ou até a OpenAlex esgotar os resultados da busca —
        o que vier primeiro.
        """
        resultados: list[dict[str, Any]] = []
        descartados_sem_abstract = 0
        cursor = "*"
        params = self._build_request_params()

        while True:
            params["cursor"] = cursor

            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                logger.error("Falha de conexão com a OpenAlex: %s", exc)
                break

            if resp.status_code in (401, 403):
                logger.error(
                    "Erro de autenticação (%s). Verifique sua API key. "
                    "Gere uma em https://openalex.org/settings/api",
                    resp.status_code,
                )
                logger.error(resp.text[:300])
                break

            if resp.status_code != 200:
                logger.error(
                    "Erro na requisição: %s - %s", resp.status_code, resp.text[:300]
                )
                break

            data = resp.json()
            works = data.get("results", [])

            if not works:
                break

            for work in works:
                artigo = self._parse_work(work)

                if self.require_abstract and not artigo["abstract"]:
                    descartados_sem_abstract += 1
                    if verbose:
                        logger.info(
                            "[descartado, sem abstract] %s", work.get("title")
                        )
                    continue

                resultados.append(artigo)

                if verbose:
                    logger.info("[%d] %s", len(resultados), work.get("title"))

                if self.max_results and len(resultados) >= self.max_results:
                    self._log_summary(resultados, descartados_sem_abstract)
                    return resultados

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(0.1)

        # A busca se esgotou antes de atingir max_results (só acontece
        # quando require_abstract=True ou quando a keyword tem poucos
        # resultados no total).
        if self.require_abstract and self.max_results and len(resultados) < self.max_results:
            logger.warning(
                "\nAviso: a busca se esgotou com apenas %d de %d abstracts "
                "solicitados (não há artigos suficientes com abstract para "
                "essas palavras-chave).",
                len(resultados),
                self.max_results,
            )

        self._log_summary(resultados, descartados_sem_abstract)
        return resultados

    def _log_summary(
        self, resultados: list[dict[str, Any]], descartados_sem_abstract: int = 0
    ) -> None:
        """
        Reporta o resultado final da busca: quantos artigos foram
        retornados, quantos vieram sem abstract (quando require_abstract
        é False, essas linhas continuam no resultado) e quantos foram
        descartados por falta de abstract (quando require_abstract é True).
        """
        if self.require_abstract:
            if descartados_sem_abstract:
                logger.info(
                    "\n%d artigos foram descartados por não terem abstract "
                    "(restrição de copyright do editor ou ausência na fonte "
                    "original).",
                    descartados_sem_abstract,
                )
            return

        if not resultados:
            return
        sem_abstract = sum(1 for r in resultados if not r["abstract"])
        if sem_abstract:
            logger.info(
                "\nAviso: %d de %d artigos vieram sem abstract "
                "(restrição de copyright do editor ou ausência na fonte original).",
                sem_abstract,
                len(resultados),
            )

    @staticmethod
    def salvar_csv(resultados: list[dict[str, Any]], caminho_saida: str) -> None:
        """Salva a lista de resultados (dicts) em um arquivo CSV."""
        if not resultados:
            logger.info("Nenhum resultado para salvar.")
            return

        campos = list(resultados[0].keys())
        with open(caminho_saida, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)

        logger.info("\n%d artigos salvos em: %s", len(resultados), caminho_saida)


# ----------------------------------------------------------------------
# Interface de linha de comando (CLI)
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openalex_downloader.py",
        description="Baixa metadados e abstracts de artigos da OpenAlex a partir de palavras-chave.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-k",
        "--api-key",
        required=True,
        help="Sua API key da OpenAlex (obrigatória). Gere em https://openalex.org/settings/api",
    )
    grupo_busca = parser.add_mutually_exclusive_group(required=True)
    grupo_busca.add_argument(
        "-w",
        "--keywords",
        nargs="+",
        help='Uma ou mais palavras-chave, combinadas com OR. Ex.: -w "soft '
        'robotics" "soft actuators". Use -Q em vez disso se precisar de '
        "lógica booleana mais complexa (AND, NOT, agrupamento).",
    )
    grupo_busca.add_argument(
        "-Q",
        "--query",
        default=None,
        help="String de busca booleana já pronta, na sintaxe da própria "
        "OpenAlex: AND / OR / NOT em MAIÚSCULAS, parênteses para agrupar, "
        'aspas duplas para frase exata. Ex.: -Q \'("REE" OR "REO") AND '
        '("rare earth" OR mining OR ore)\'',
    )
    parser.add_argument(
        "-e",
        "--email",
        default=None,
        help="Seu e-mail (opcional, recomendado para o polite pool da OpenAlex).",
    )
    parser.add_argument(
        "-m",
        "--max-results",
        type=int,
        default=None,
        help="Baixa apenas os N resultados mais relevantes. "
        "Se omitido, baixa todos os resultados encontrados.",
    )
    parser.add_argument(
        "-p",
        "--per-page",
        type=int,
        default=200,
        help="Resultados por página (máximo permitido pela API: 200).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="openalex_resultados.csv",
        help="Caminho do arquivo CSV de saída.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Não imprime o progresso artigo a artigo durante a busca.",
    )
    parser.add_argument(
        "--open-access",
        action="store_true",
        help="Retorna apenas artigos em acesso aberto (Open Access).",
    )
    parser.add_argument(
        "--require-abstract",
        action="store_true",
        help="Descarta artigos sem abstract e continua buscando até juntar "
        "--max-results artigos COM abstract (em vez de --max-results "
        "artigos no total).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        downloader = OpenAlexDownloader(
            api_key=args.api_key,
            keywords=args.keywords,
            query=args.query,
            email=args.email,
            max_results=args.max_results,
            per_page=args.per_page,
            open_access_only=args.open_access,
            require_abstract=args.require_abstract,
        )
    except ValueError as e:
        parser.error(str(e))
        return

    logger.info("Buscando artigos sobre: %s...\n", args.query or args.keywords)
    artigos = downloader.buscar(verbose=not args.quiet)
    downloader.salvar_csv(artigos, args.output)


if __name__ == "__main__":
    main()