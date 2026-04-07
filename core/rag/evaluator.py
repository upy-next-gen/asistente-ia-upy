import asyncio

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import llm_factory

from core.config import settings
from core.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    def __init__(self):
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        llm = llm_factory(settings.LLM_MODEL, client=client)
        self._faithfulness = Faithfulness(llm=llm)

    def rerank(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return rows

        sorted_rows = sorted(
            rows,
            key=lambda r: float(r.get("similarity", 0)),
            reverse=True,
        )

        filtered = [
            r for r in sorted_rows
            if float(r.get("similarity", 0)) >= settings.RAGAS_MIN_RELEVANCY
        ]
        result = filtered[:settings.RAGAS_TOP_FINAL]

        if not result and sorted_rows:
            result = [sorted_rows[0]]
            logger.warning(
                "Ningun chunk supero el umbral %.2f. Usando el mejor disponible.",
                settings.RAGAS_MIN_RELEVANCY,
            )

        logger.info(
            "Reranking: %s/%s chunks pasaron el filtro (umbral=%.2f).",
            len(result),
            len(rows),
            settings.RAGAS_MIN_RELEVANCY,
        )
        return result

    async def evaluate(
        self,
        query: str,
        response: str,
        rows: list[dict],
    ) -> dict[str, float]:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=[r.get("content", "") for r in rows],
        )

        faith = await self._faithfulness.single_turn_ascore(sample)

        logger.info("RAGAS — faithfulness: %.2f", faith)
        return {"faithfulness": float(faith)}
    