import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the parent directory's .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)
model_name = os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")
rag_client = default_rag_client(llm_client=openai_client, logdir="evals/logs")
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


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
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
