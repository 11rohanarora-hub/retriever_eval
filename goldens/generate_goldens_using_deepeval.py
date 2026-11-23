
import os
import glob
import json
import random

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# --------------------------------------------------
# 2. Custom OpenRouter model for DeepEval
# --------------------------------------------------


class OpenRouterModel(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-4.1-mini"):
        self.model_name = model

        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=4096
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
# 3. Load, clean and chunk VTT files
# --------------------------------------------------

def load_chunks():

    texts = []

    for path in glob.glob("data/*.vtt"):

        with open(path, "r", encoding="utf-8") as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
                and line.strip() != "WEBVTT"
                and "-->" not in line
            ]

        cleaned_text = " ".join(lines)

        texts.append({
            "source": os.path.basename(path),
            "text": cleaned_text
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks_with_source = []

    for item in texts:

        chunks = splitter.split_text(item["text"])

        for chunk in chunks:
            chunks_with_source.append({
                "text": chunk,
                "source": item["source"]
            })

    return chunks_with_source


# --------------------------------------------------
# 4. Generate chunks
# --------------------------------------------------

chunks = load_chunks()

if not chunks:
    raise ValueError("No VTT files or text chunks found in data/")

print(f"Total chunks available: {len(chunks)}")


# --------------------------------------------------
# 5. Randomly sample chunks
# --------------------------------------------------

random.seed(42)

sample = random.sample(
    chunks,
    min(1, len(chunks))
)

# DeepEval expects a list of lists.
# Each context contains one chunk.

contexts = [
    [item["text"]]
    for item in sample
]


# --------------------------------------------------
# 6. Initialize OpenRouter + DeepEval Synthesizer
# --------------------------------------------------

openrouter_model = OpenRouterModel(
    model="openai/gpt-4.1-mini"
)

synthesizer = Synthesizer(
    model=openrouter_model
)


# --------------------------------------------------
# 7. Generate Goldens
# --------------------------------------------------

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1
)


# --------------------------------------------------
# 8. Convert Goldens into custom JSON schema
# --------------------------------------------------

rows = []

for i, golden in enumerate(goldens, 1):

    rows.append({
        "id": f"g{i:03d}",
        "query": golden.input,
        "ideal_answer": golden.expected_output,
        "source": sample[i - 1]["source"]
    })


# --------------------------------------------------
# 9. Save generated Goldens
# --------------------------------------------------

os.makedirs("goldens", exist_ok=True)

output_path = "goldens/retriever_deepeval_goldens.json"

with open(output_path, "w", encoding="utf-8") as f:

    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


# --------------------------------------------------
# 10. Print summary
# --------------------------------------------------

print(f"\nSuccessfully generated {len(rows)} Goldens.")

print(f"Output saved to: {output_path}")

print("\nIMPORTANT: Review every Golden before using it.")

print("Check:")
print("1. Is the question answerable from the source context?")
print("2. Is the ideal answer factually correct?")
print("3. Does the source filename match the original transcript?")
