#!/usr/bin/env python3

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cai.util import COST_TRACKER, calculate_model_cost


def setup_test():
    """Clear the pricing cache before each test"""
    COST_TRACKER.model_pricing_cache.clear()


def test_local_models_return_zero_cost():
    """Test that local models return zero cost"""
    setup_test()
    
    local_models = [
        "qwen3:14b",
        "qwen3:32b", 
        "qwen2.5:14b",
        "qwen2.5:7b",
        "qwen2.5:72b",
        "llama3.1:8b",
        "llama3.1:70b",
        "mistral:7b",
        "mistral:latest",
        "codellama:13b",
        "ollama/llama3.1",
        "ollama/qwen2.5",
        "deepseek-coder:6.7b",
        "phi3:mini",
        "gemma:7b",
        "vicuna:13b",
        "alpaca:7b",
        "orca-mini:3b",
        "neural-chat:7b",
        "starling-lm:7b",
        "zephyr:7b",
        "openchat:7b",
        "wizard-coder:15b",
        "sqlcoder:7b",
        "magicoder:7b",
        "dolphin-mixtral:8x7b",
        "nous-hermes2:10.7b",
        "yi:34b",
        "qwq:32b",
        "alias01:14b",
        "alias01:14b-ctx-32000",
        "alias00:14b"
    ]
    
    failed_models = []
    
    for model in local_models:
        with patch('requests.get') as mock_get:
            # Mock LiteLLM API to return empty response (model not found)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}  # Empty response, model not found
            mock_get.return_value = mock_response
            
            pricing = COST_TRACKER.get_model_pricing(model)
            cost = calculate_model_cost(model, 100, 50)
            
            if pricing != (0.0, 0.0):
                failed_models.append(f"Model {model} should have zero pricing, got {pricing}")
            if cost != 0.0:
                failed_models.append(f"Model {model} should have zero cost, got {cost}")
    
    if failed_models:
        print("FAILED: test_local_models_return_zero_cost")
        for failure in failed_models:
            print(f"  - {failure}")
        return False
    else:
        print("PASSED: test_local_models_return_zero_cost")
        return True


def test_paid_models_return_nonzero_cost():
    """Test that known paid models return non-zero cost"""
    setup_test()
    
    paid_models_with_expected_pricing = {
        "gpt-4": {"input_cost_per_token": 0.00003, "output_cost_per_token": 0.00006},
        "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        "claude-3-sonnet-20240229": {"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015},
        "claude-3-5-sonnet-20241022": {"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015}
    }
    
    failed_models = []
    
    for model, expected_pricing in paid_models_with_expected_pricing.items():
        with patch('requests.get') as mock_get:
            # Mock LiteLLM API to return pricing for paid models
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                model: expected_pricing
            }
            mock_get.return_value = mock_response
            
            pricing = COST_TRACKER.get_model_pricing(model)
            cost = calculate_model_cost(model, 100, 50)
            
            if not (pricing[0] > 0 or pricing[1] > 0):
                failed_models.append(f"Model {model} should have non-zero pricing, got {pricing}")
            if not (cost > 0):
                failed_models.append(f"Model {model} should have non-zero cost, got {cost}")
    
    if failed_models:
        print("FAILED: test_paid_models_return_nonzero_cost")
        for failure in failed_models:
            print(f"  - {failure}")
        return False
    else:
        print("PASSED: test_paid_models_return_nonzero_cost")
        return True


def test_private_model_alias0_with_pricing_json():
    """Test that alias0 works correctly when defined in pricing.json"""
    setup_test()
    
    # Create a pricing.json with alias0 configuration
    pricing_config = {
        "alias0": {
            "max_tokens": 128000,
            "max_input_tokens": 200000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 5e-05,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_vision": True
        }
    }
    
    try:
        # Mock the file reading to simulate pricing.json with alias0
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value = mock_file
                mock_file.read.return_value = json.dumps(pricing_config)
                mock_open.return_value = mock_file
                
                # Mock json.load to return our config
                with patch('json.load', return_value=pricing_config):
                    pricing = COST_TRACKER.get_model_pricing("alias0")
                    cost = calculate_model_cost("alias0", 100, 50)
                    
                    expected_pricing = (5e-06, 5e-05)
                    expected_cost = 100 * 5e-06 + 50 * 5e-05  # 0.0030
                    
                    if pricing != expected_pricing:
                        print(f"FAILED: test_private_model_alias0_with_pricing_json")
                        print(f"  - alias0 should have pricing {expected_pricing}, got {pricing}")
                        return False
                    
                    if abs(cost - expected_cost) >= 1e-10:
                        print(f"FAILED: test_private_model_alias0_with_pricing_json")
                        print(f"  - alias0 should have cost {expected_cost}, got {cost}")
                        return False
        
        print("PASSED: test_private_model_alias0_with_pricing_json")
        return True
        
    except Exception as e:
        print(f"FAILED: test_private_model_alias0_with_pricing_json")
        print(f"  - Exception: {e}")
        return False


def test_reset_cost_for_local_model():
    """Test the reset_cost_for_local_model function"""
    setup_test()
    
    failed_tests = []
    
    # Test with a free model
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # Model not found, will return (0.0, 0.0)
        mock_get.return_value = mock_response
        
        result = COST_TRACKER.reset_cost_for_local_model("qwen3:14b")
        if result != True:
            failed_tests.append("qwen3:14b should be identified as a free model")
    
    # Test with a paid model
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "gpt-4": {
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006
            }
        }
        mock_get.return_value = mock_response
        
        result = COST_TRACKER.reset_cost_for_local_model("gpt-4")
        if result != False:
            failed_tests.append("gpt-4 should not be identified as a free model")
    
    if failed_tests:
        print("FAILED: test_reset_cost_for_local_model")
        for failure in failed_tests:
            print(f"  - {failure}")
        return False
    else:
        print("PASSED: test_reset_cost_for_local_model")
        return True


def run_all_tests():
    """Run all tests and report results"""
    print("Running pricing tests...")
    print("=" * 50)
    
    tests = [
        test_local_models_return_zero_cost,
        test_paid_models_return_nonzero_cost,
        test_private_model_alias0_with_pricing_json,
        test_reset_cost_for_local_model
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  - Exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! ✅")
        return True
    else:
        print("Some tests failed! ❌")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 