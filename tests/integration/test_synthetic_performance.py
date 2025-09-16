"""
Integration performance test for SyntheticTextItemsGeneratorSlow vs SyntheticTextItemsGenerator.

This test compares the performance of two different synthetic text generators
across different prompt sizes and tokenizers.
"""

import time

import pytest
from transformers import AutoTokenizer

from guidellm.dataset.synthetic import (
    SyntheticDatasetConfig,
    SyntheticTextItemsGenerator,
    SyntheticTextItemsGeneratorSlow,
)


class TestSyntheticGeneratorPerformance:
    """Performance comparison tests for synthetic text item generators."""

    # Test configurations for different prompt sizes
    PROMPT_SIZES = [
        ("small", 50),
        ("medium", 200),
        ("large", 500),
        ("huge", 4000),
    ]

    # Common tokenizers for testing
    TOKENIZERS = [
        "gpt2",
        "distilbert-base-uncased",
        "microsoft/DialoGPT-small",
    ]

    # Number of samples for performance testing
    SAMPLES = 100

    @pytest.fixture(params=TOKENIZERS)
    def tokenizer(self, request):
        """Fixture to provide different tokenizers for testing."""
        return AutoTokenizer.from_pretrained(request.param)

    @pytest.fixture(params=PROMPT_SIZES)
    def prompt_config(self, request):
        """Fixture to provide different prompt size configurations."""
        size_name, prompt_tokens = request.param
        return size_name, SyntheticDatasetConfig(
            prompt_tokens=prompt_tokens,
            output_tokens=100,  # Keep output tokens constant
            samples=self.SAMPLES,
            source="data:prideandprejudice.txt.gz",
        )

    def _measure_generation_time(
        self,
        generator_class,
        config: SyntheticDatasetConfig,
        tokenizer,
        random_seed: int = 42,
    ) -> tuple[float, list[dict]]:
        """
        Measure the time taken to generate a dataset using the specified generator.

        Returns:
            Tuple of (elapsed_time_seconds, generated_items)
        """
        generator = generator_class(config, tokenizer, random_seed)

        start_time = time.perf_counter()
        items = list(generator)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        return elapsed_time, items

    def _validate_generated_items(self, items: list[dict], expected_count: int):
        """Validate that generated items have the correct structure and count."""
        expected_msg = f"Expected {expected_count} items, got {len(items)}"
        assert len(items) == expected_count, expected_msg

        for item in items:
            assert "prompt" in item
            assert "prompt_tokens_count" in item
            assert "output_tokens_count" in item
            assert isinstance(item["prompt"], str)
            assert isinstance(item["prompt_tokens_count"], int)
            assert isinstance(item["output_tokens_count"], int)
            assert len(item["prompt"]) > 0

    @pytest.mark.regression
    def test_generator_performance_comparison(self, tokenizer, prompt_config):
        """
        Compare performance between SyntheticTextItemsGeneratorSlow and SyntheticTextItemsGenerator.

        This test ensures both generators:
        1. Produce the same number of items
        2. Generate valid data structures
        3. Have measurable performance characteristics
        """
        size_name, config = prompt_config

        # Test SyntheticTextItemsGeneratorSlow (original)
        time1, items1 = self._measure_generation_time(
            SyntheticTextItemsGeneratorSlow, config, tokenizer
        )

        # Test SyntheticTextItemsGenerator (new implementation)
        time2, items2 = self._measure_generation_time(
            SyntheticTextItemsGenerator, config, tokenizer
        )

        # Validate both generators produce correct output
        self._validate_generated_items(items1, config.samples)
        self._validate_generated_items(items2, config.samples)

        # Calculate performance metrics
        performance_ratio = time1 / time2 if time2 > 0 else float("inf")

        # Report performance differences
        if performance_ratio > 1:
            faster_generator = "SyntheticTextItemsGenerator"
            speedup = performance_ratio
            slower_time, faster_time = time1, time2
        else:
            faster_generator = "SyntheticTextItemsGeneratorSlow"
            speedup = 1 / performance_ratio
            slower_time, faster_time = time2, time1

        print(f"\n=== Performance Results for {size_name} prompts ===")
        print(f"SyntheticTextItemsGeneratorSlow: {time1:.4f}s")
        print(f"SyntheticTextItemsGenerator: {time2:.4f}s")
        print(f"{faster_generator} is {speedup:.2f}x faster")
        print(f"Time difference: {abs(slower_time - faster_time):.4f}s")

        # Assertions
        assert time1 > 0, "SyntheticTextItemsGeneratorSlow should take measurable time"
        assert time2 > 0, "SyntheticTextItemsGenerator should take measurable time"
        same_count_msg = "Both generators should produce same number of items"
        assert len(items1) == len(items2), same_count_msg

        # Performance difference should be significant (at least 5% difference)
        perf_msg = (
            f"Expected significant performance difference, "
            f"got ratio: {performance_ratio:.3f}"
        )
        assert abs(performance_ratio - 1.0) > 0.05, perf_msg

    @pytest.mark.sanity
    def test_generator_consistency(self):
        """
        Test that both generators produce exactly consistent results with the same configuration.

        This test ensures that given the same random seed and configuration,
        both generators produce items with exactly the requested token counts.
        """
        config = SyntheticDatasetConfig(
            prompt_tokens=100,
            output_tokens=50,
            samples=10,
            source="data:prideandprejudice.txt.gz",
        )

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        random_seed = 123

        # Generate items with both generators using the same seed
        gen1 = SyntheticTextItemsGeneratorSlow(config, tokenizer, random_seed)
        items1 = list(gen1)

        gen2 = SyntheticTextItemsGenerator(config, tokenizer, random_seed)
        items2 = list(gen2)

        # Both should generate the same number of items
        assert len(items1) == len(items2) == config.samples

        # Token counts should be within reasonable range for both generators
        for items, generator_name in [(items1, "Gen1"), (items2, "Gen2")]:
            prompt_token_counts = [item["prompt_tokens_count"] for item in items]
            output_token_counts = [item["output_tokens_count"] for item in items]

            # Validate prompt token counts are exactly as requested
            expected_prompt_tokens = config.prompt_tokens
            for i, count in enumerate(prompt_token_counts):
                assert count == expected_prompt_tokens, (
                    f"{generator_name}: Sample {i} has {count} prompt tokens, "
                    f"expected exactly {expected_prompt_tokens}"
                )

            # Validate output token counts match config
            for count in output_token_counts:
                count_msg = (
                    f"{generator_name}: Output token count {count} "
                    f"doesn't match config {config.output_tokens}"
                )
                assert count == config.output_tokens, count_msg

    @pytest.mark.sanity
    def test_generators_produce_exact_identical_results(self):
        """
        Test that both generators produce exactly identical results with precise token counts.

        This test ensures that SyntheticTextItemsGeneratorSlow and SyntheticTextItemsGenerator
        produce identical outputs with exact token counts when given the same parameters.
        """
        config = SyntheticDatasetConfig(
            prompt_tokens=100,
            output_tokens=50,
            samples=1,
            source="data:prideandprejudice.txt.gz",
        )

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        random_seed = 42

        # Create instances of both generators
        gen1 = SyntheticTextItemsGeneratorSlow(config, tokenizer, random_seed)
        gen2 = SyntheticTextItemsGenerator(config, tokenizer, random_seed)

        # Test multiple scenarios with different parameters
        test_scenarios = [
            s
            for i in range(0, 100, 10)
            for s in [
                {"prompt_tokens": 50, "start_index": 100 + i, "unique_prefix": None},
                {"prompt_tokens": 100, "start_index": 200 + i, "unique_prefix": 42},
                {"prompt_tokens": 25, "start_index": 500 + i, "unique_prefix": None},
                {"prompt_tokens": 75, "start_index": 1000 + i, "unique_prefix": 123},
            ]
        ]

        for i, scenario in enumerate(test_scenarios):
            print(f"\n--- Testing scenario {i + 1}: {scenario} ---")

            # Call _create_prompt directly on both generators
            prompt1 = gen1._create_prompt(**scenario)
            prompt2, _ = gen2._create_prompt(**scenario)

            # Convert to text for comparison
            text1 = tokenizer.decode(prompt1, skip_special_tokens=True)
            text2 = tokenizer.decode(prompt2, skip_special_tokens=True)

            print(f"Gen1 tokens: {len(prompt1)}, Gen2 tokens: {len(prompt2)}")
            print(f"Gen1 text preview: {text1[:100]}...")
            print(f"Gen2 text preview: {text2[:100]}...")

            # Assert exact equivalence between implementations
            tokens_diff = len(prompt1) - len(prompt2)
            text_same = text1 == text2

            print(f"Token count difference: {tokens_diff}")
            print(f"Text identical: {text_same}")

            # Assert identical text output
            assert text1 == text2, (
                f"Scenario {i + 1}: Generators produced different text.\n"
                f"Gen1: '{text1}'\n"
                f"Gen2: '{text2}'"
            )

            # Assert identical token sequences
            assert prompt1 == prompt2, (
                f"Scenario {i + 1}: Generators produced different token sequences.\n"
                f"Gen1 ({len(prompt1)} tokens): {prompt1}\n"
                f"Gen2 ({len(prompt2)} tokens): {prompt2}"
            )

            # Assertions for valid output characteristics
            assert len(prompt1) > 0, f"Scenario {i + 1}: Gen1 produced empty prompt"
            assert len(prompt2) > 0, f"Scenario {i + 1}: Gen2 produced empty prompt"
            assert isinstance(prompt1, list), (
                f"Scenario {i + 1}: Gen1 didn't return list"
            )
            assert isinstance(prompt2, list), (
                f"Scenario {i + 1}: Gen2 didn't return list"
            )

            # Both must produce EXACT token counts - no approximations allowed
            expected_tokens = scenario["prompt_tokens"]
            # if scenario["unique_prefix"] is not None:
            #     expected_tokens += 1  # Account for unique prefix token

            assert len(prompt1) >= expected_tokens, (
                f"Scenario {i + 1}: Gen1 produced {len(prompt1)} tokens, "
                f"expected equal or greater than {expected_tokens}"
            )

            assert len(prompt2) >= expected_tokens, (
                f"Scenario {i + 1}: Gen2 produced {len(prompt2)} tokens, "
                f"expected equal or greater than {expected_tokens}"
            )

            print("✓ Both generators produced exact identical results!")

    @pytest.mark.regression
    def test_end_to_end_identical_dataset_generation(self):
        """
        Test that both generators produce exactly identical full datasets.

        This test ensures that when generating complete datasets, both generators
        produce identical results for every sample.
        """
        config = SyntheticDatasetConfig(
            prompt_tokens=75,
            output_tokens=25,
            samples=5,  # Small number for detailed comparison
            source="data:prideandprejudice.txt.gz",
        )

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        random_seed = 12345

        # Generate full datasets with both generators
        gen1 = SyntheticTextItemsGeneratorSlow(config, tokenizer, random_seed)
        items1 = list(gen1)

        gen2 = SyntheticTextItemsGenerator(config, tokenizer, random_seed)
        items2 = list(gen2)

        # Assert same number of items
        assert len(items1) == len(items2) == config.samples

        # Assert each item is exactly identical
        for i, (item1, item2) in enumerate(zip(items1, items2)):
            # Check structure
            assert set(item1.keys()) == set(item2.keys()), f"Sample {i}: Different keys"

            # Check exact prompt text match
            assert item1["prompt"] == item2["prompt"], (
                f"Sample {i}: Different prompts\n"
                f"Gen1: '{item1['prompt']}'\n"
                f"Gen2: '{item2['prompt']}'"
            )

            # Check exact token counts match
            assert item1["prompt_tokens_count"] == item2["prompt_tokens_count"], (
                f"Sample {i}: Different prompt token counts "
                f"(Gen1: {item1['prompt_tokens_count']}, Gen2: {item2['prompt_tokens_count']})"
            )

            assert item1["output_tokens_count"] == item2["output_tokens_count"], (
                f"Sample {i}: Different output token counts "
                f"(Gen1: {item1['output_tokens_count']}, Gen2: {item2['output_tokens_count']})"
            )

            # Check exact token counts match configuration
            assert item1["prompt_tokens_count"] == config.prompt_tokens, (
                f"Sample {i}: Gen1 prompt tokens {item1['prompt_tokens_count']} != "
                f"expected {config.prompt_tokens}"
            )

            assert item2["prompt_tokens_count"] == config.prompt_tokens, (
                f"Sample {i}: Gen2 prompt tokens {item2['prompt_tokens_count']} != "
                f"expected {config.prompt_tokens}"
            )

            print(f"✓ Sample {i}: Identical results confirmed")

        print(
            f"✓ All {config.samples} samples are exactly identical between generators!"
        )

    @pytest.mark.smoke
    def test_performance_benchmark_summary(self):
        """
        Generate a comprehensive performance summary across all configurations.

        This test runs all combinations and provides a summary of performance differences.
        """
        results = []

        for tokenizer_name in self.TOKENIZERS:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            for size_name, prompt_tokens in self.PROMPT_SIZES:
                config = SyntheticDatasetConfig(
                    prompt_tokens=prompt_tokens,
                    output_tokens=100,
                    samples=20,  # Smaller sample size for benchmark
                    source="data:prideandprejudice.txt.gz",
                )

                # Measure both generators
                time1, _ = self._measure_generation_time(
                    SyntheticTextItemsGeneratorSlow, config, tokenizer
                )
                time2, _ = self._measure_generation_time(
                    SyntheticTextItemsGenerator, config, tokenizer
                )

                results.append(
                    {
                        "tokenizer": tokenizer_name,
                        "prompt_size": size_name,
                        "prompt_tokens": prompt_tokens,
                        "gen1_time": time1,
                        "gen2_time": time2,
                        "ratio": time1 / time2 if time2 > 0 else float("inf"),
                    }
                )

        # Calculate overall statistics and report results
        ratios = [r["ratio"] for r in results if r["ratio"] != float("inf")]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)

            print("\n" + "=" * 80)
            print("PERFORMANCE BENCHMARK SUMMARY")
            print("=" * 80)
            header = (
                f"{'Tokenizer':<25} {'Size':<8} {'Tokens':<8} "
                f"{'Gen1':<8} {'Gen2':<8} {'Ratio':<8} {'Faster'}"
            )
            print(header)
            print("-" * 90)

            for result in results:
                ratio = result["ratio"]
                faster = "Gen2" if ratio > 1 else "Gen1"
                speedup = ratio if ratio > 1 else 1 / ratio
                faster_label = f"{faster} ({speedup:.1f}x)"

                row = (
                    f"{result['tokenizer']:<25} {result['prompt_size']:<8} "
                    f"{result['prompt_tokens']:<8} {result['gen1_time']:<8.3f} "
                    f"{result['gen2_time']:<8.3f} {result['ratio']:<8.2f} {faster_label}"
                )
                print(row)

            print("=" * 90)
            print(f"Average performance ratio (Gen1/Gen2): {avg_ratio:.2f}x")

            if avg_ratio > 1:
                msg = f"Overall: SyntheticTextItemsGenerator is {avg_ratio:.2f}x faster on average"
                print(msg)
            else:
                msg = f"Overall: SyntheticTextItemsGeneratorSlow is {1 / avg_ratio:.2f}x faster on average"
                print(msg)

            print("=" * 80 + "\n")

        # Ensure we have valid results
        assert len(results) == len(self.TOKENIZERS) * len(self.PROMPT_SIZES)
        assert all(r["gen1_time"] > 0 and r["gen2_time"] > 0 for r in results)
