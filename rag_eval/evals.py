import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the parent directory's .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from ragas import Dataset, experiment, SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness

# Add the project root to the path so we can import src module
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.grounded_answer import generate_answer

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)
model_name = os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")
llm = llm_factory(model_name, client=openai_client)


def load_dataset():
    dataset = Dataset(
        name="project_eval_set",
        backend="local/csv",
        root_dir="evals",
    )

    questions_path = Path(__file__).parent.parent / "data/eval/questions.jsonl"
    
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            # Map your project fields to Ragas fields
            # We use 'expected_doc_id' as a placeholder for 'grading_notes' for now
            row = {
                "question": sample["question"], 
                "grading_notes": sample.get("expected_doc_id", "N/A"),
                "id": sample.get("id")
            }
            dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


# Initialize the Faithfulness metric
faithfulness = Faithfulness(llm=llm)


@experiment()
async def run_experiment(row):
    # Call our real orchestrator
    result = generate_answer(
        question=row["question"],
        mode="llm",  # We want to test the LLM grounded mode
        index_path=PROJECT_ROOT / "data/processed/retrieval_index.json"
    )

    # Extract context strings from citations for the Faithfulness check
    contexts = [c.cited_text for c in result.citations]
    
    # If no citations were made, we still need to provide context if available
    if not contexts and result.insufficient_evidence is False:
        # Fallback to a placeholder if retrieval happened but citations didn't
        contexts = ["Context was retrieved but no specific citations were made."]

    # 1. Standardize the data into a SingleTurnSample
    sample = SingleTurnSample(
        user_input=row["question"],
        response=result.answer,
        retrieved_contexts=contexts
    )

    # 2. Use the new async scoring method for a single sample
    score = await faithfulness.single_ascore(sample)

    experiment_view = {
        **row,
        "response": result.answer,
        "faithfulness_score": score.value,
        "insufficient_evidence": result.insufficient_evidence,
        "num_citations": len(result.citations)
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
