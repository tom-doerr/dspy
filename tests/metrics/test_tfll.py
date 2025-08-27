"""Test TFLL metric implementation."""

import pytest
from unittest.mock import Mock
from dspy.metrics.tfll import TFLLMetric, _extract_label_prompt_tokens, _avg_margin


class TestTFLLMetric:
    """Test the TFLL metric for DSPy."""
    
    def test_tfll_metric_creation(self):
        """Test creating a TFLL metric instance."""
        raw_chat = Mock(return_value={"choices": []})
        metric = TFLLMetric(
            raw_chat=raw_chat,
            model="test/model"
        )
        assert metric.model == "test/model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])