
import os
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from langchain_openai import ChatOpenAI
from src.reranker import RerankingRetriever


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# --------------------------------------------------
# 2. CUSTOM OPENROUTER MODEL FOR DEEPEVAL
# --------------------------------------------------

class OpenRouterModel(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-4.1-mini"):
        self.model_name = model

        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=1024,
        )

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return self.model_name


# --------------------------------------------------
# 3. CONFIGURATION
# --------------------------------------------------

GOLDEN_PATH = "goldens/retriever_goldens_using_llm.json"

THRESHOLD = 0.7

JUDGE_MODEL = OpenRouterModel(
    model="openai/gpt-4.1-mini"
)


# --------------------------------------------------
# 4. LOAD GOLDEN DATASET
# --------------------------------------------------

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)


# --------------------------------------------------
# 5. RUN RETRIEVER
# --------------------------------------------------

retriever = RerankingRetriever()

test_cases = []

for g in goldens:

    retrieved = retriever.invoke(g["query"])

    retrieval_context = [
        doc.page_content for doc in retrieved
    ]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="Generator not evaluated in this run",
        )
    )


# --------------------------------------------------
# 6. DEFINE EVALUATION METRICS
# --------------------------------------------------

metrics = [

    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),

    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),

]


# --------------------------------------------------
# 7. RUN EVALUATION
# --------------------------------------------------

evaluate(
    test_cases=test_cases,
    metrics=metrics,

    async_config=AsyncConfig(
        run_async=False
    ),

    hyperparameters={
        "retriever": "reranked",
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": "openai/gpt-4.1-mini via OpenRouter",
        "golden_set": GOLDEN_PATH,
    },
)
