
# eval_retriever.py

import os
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from langchain_openai import ChatOpenAI

from src.retriever import build_retriever
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. "
        "Please add it to your .env file."
    )


# --------------------------------------------------
# 2. OPENROUTER CUSTOM MODEL FOR DEEPEVAL
# --------------------------------------------------

class OpenRouterModel(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-4o-mini"):
        self.model_name = model

        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=1096,
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
# 3. EVALUATION CONFIGURATION
# --------------------------------------------------

GOLDEN_PATH = "goldens/retriever_goldens_using_llm.json"

THRESHOLD = 0.7

# Initialize OpenRouter judge model
JUDGE_MODEL = OpenRouterModel(
    model="openai/gpt-4o-mini"
)


# --------------------------------------------------
# 4. RUN RETRIEVER EVALUATION
# --------------------------------------------------

def run(retriever):

    # Step 1: Load the golden dataset
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    # Step 2: Retrieve documents for each question
    for g in goldens:

        retrieved = retriever.invoke(g["query"])

        retrieval_context = [
            doc.page_content for doc in retrieved
        ]

        # Step 3: Create DeepEval test case
        test_case = LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="Generator not evaluated in this run",
        )

        test_cases.append(test_case)

    # Step 4: Define evaluation metrics
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

    # # Step 5: Run evaluation
    # result = evaluate(
    #     test_cases=test_cases,
    #     metrics=metrics,

    #     hyperparameters={
    #         "retriever": "base_k5",
    #         "embedding_model": "text-embedding-3-large",
    #         "chunk_size": 1000,
    #         "chunk_overlap": 150,
    #         "top_k": 3,
    #         "judge_model": "openai/gpt-4o-mini via OpenRouter",
    #         "golden_set": GOLDEN_PATH,
    #     }
    # )

    # Step 6: Summarize results
    # return summarize_by_metric(result)


    # Run each metric separately to reduce simultaneous API requests

    hyperparameters = {
        "retriever": "reranker",
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 3,
        "judge_model": "openai/gpt-4o-mini via OpenRouter",
        "golden_set": GOLDEN_PATH,
    }

    final_summary = {}

    for metric in metrics:

        print(f"\nRunning metric: {metric.__class__.__name__}")

        result = evaluate(
            test_cases=test_cases,
            metrics=[metric],
            hyperparameters=hyperparameters,
        )

        metric_summary = summarize_by_metric(result)
        final_summary.update(metric_summary)

    return final_summary



# --------------------------------------------------
# 5. STANDALONE EXECUTION
# --------------------------------------------------

def run_local():

    retriever = build_retriever()

    return run(retriever)


if __name__ == "__main__":

    print_summary(
        "retriever",
        run_local()
    )
