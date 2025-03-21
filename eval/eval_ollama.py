from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models import OllamaModel

# Define benchmark with specific tasks and shots
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

#Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=OllamaModel(model_id="llama3.2"))
print(benchmark.overall_score)