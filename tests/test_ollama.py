import unittest
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
)

simple_test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris."
)

class TestOllamaModelLive(unittest.TestCase):

    def test_generate_live(self):
        prompt = "What is the capital of France?"
        model = OllamaModel(
            model_id="llama3.2",
        )
        result = model.generate(prompt)
        # Basic assertions to verify response
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)  # Response should have some content

    def test_different_prompts(self):
        """Test different prompts to ensure varied responses."""
        prompts = [
            "Introduce artificial intelligence",
            "Introduce Pythagorean theorem",
            "Write a short poem about nature."
        ]
        model = OllamaModel(
            model_id="llama3.2",
        )
        responses = [model.generate(prompt) for prompt in prompts]
        
        # Verify all responses are different
        self.assertNotEqual(responses[0], responses[1])
        self.assertNotEqual(responses[1], responses[2])
        self.assertNotEqual(responses[0], responses[2])

    def test_simple_evaluation(self):
        """Test simple evaluation with AnswerRelevancyMetric."""
        model = OllamaModel(
            model_id="llama3.2",
        )

        metric = AnswerRelevancyMetric(model=model, threshold=0.8)
        assert_test(simple_test_case, [metric])

if __name__ == "__main__":
    unittest.main()
