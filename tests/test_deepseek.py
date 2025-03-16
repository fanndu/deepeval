import unittest
import os
from deepeval.models import DeepSeekModel

# Skip these tests if DEEPSEEK_API_KEY is not set
skip_live_tests =  False

@unittest.skipIf(skip_live_tests, "DEEPSEEK_API_KEY not set, skipping live tests")
class TestDeepSeekModelLive(unittest.TestCase):
    """Live test cases for the DeepSeekModel class that require an API key."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepSeekModel(api_key="NA")

    def test_generate_live(self):
        """Test the generate method with a live API call."""
        prompt = "What is the capital of France?"
        result = self.model.generate(prompt)
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
        
        responses = [self.model.generate(prompt) for prompt in prompts]
        
        # Verify all responses are different
        self.assertNotEqual(responses[0], responses[1])
        self.assertNotEqual(responses[1], responses[2])
        self.assertNotEqual(responses[0], responses[2])


if __name__ == "__main__":
    unittest.main()
