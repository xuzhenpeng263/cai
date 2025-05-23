#!/usr/bin/env python3

import pathlib
import json
import requests
from src.cai.util import COST_TRACKER, get_model_name

def debug_pricing_flow():
    """Debug step by step the get_model_pricing flow"""
    
    model_name = 'qwen3:14b'
    print(f"Debugging pricing flow for: {model_name}")
    
    # Step 1: Standardize model name
    standardized = get_model_name(model_name)
    print(f"1. Standardized model name: {standardized}")
    
    # Step 2: Check cache
    if standardized in COST_TRACKER.model_pricing_cache:
        cached = COST_TRACKER.model_pricing_cache[standardized]
        print(f"2. Found in cache: {cached}")
        return cached
    else:
        print("2. Not found in cache")
    
    # Step 3: Check local pricing.json
    print("3. Checking local pricing.json...")
    try:
        pricing_path = pathlib.Path("pricing.json")
        if pricing_path.exists():
            print("   pricing.json exists")
            with open(pricing_path, "r", encoding="utf-8") as f:
                local_pricing = json.load(f)
                print(f"   Content: {local_pricing}")
                pricing_info = local_pricing.get("alias0", {})
                input_cost = pricing_info.get("input_cost_per_token", 0)
                output_cost = pricing_info.get("output_cost_per_token", 0)
                print(f"   Extracted pricing: input={input_cost}, output={output_cost}")
                if input_cost or output_cost:
                    print(f"   Would return: ({input_cost}, {output_cost})")
                    return (input_cost, output_cost)
        else:
            print("   pricing.json does not exist")
    except Exception as e:
        print(f"   Error reading pricing.json: {e}")
    
    # Step 4: Check LiteLLM API
    print("4. Checking LiteLLM API...")
    LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    try:
        response = requests.get(LITELLM_URL, timeout=2)
        if response.status_code == 200:
            model_pricing_data = response.json()
            pricing_info = model_pricing_data.get(standardized, {})
            input_cost_per_token = pricing_info.get("input_cost_per_token", 0)
            output_cost_per_token = pricing_info.get("output_cost_per_token", 0)
            print(f"   LiteLLM response for {standardized}: {pricing_info}")
            print(f"   Extracted: input={input_cost_per_token}, output={output_cost_per_token}")
            if input_cost_per_token or output_cost_per_token:
                print(f"   Would return: ({input_cost_per_token}, {output_cost_per_token})")
                return (input_cost_per_token, output_cost_per_token)
        else:
            print(f"   LiteLLM API returned status: {response.status_code}")
    except Exception as e:
        print(f"   Error fetching from LiteLLM: {e}")
    
    # Step 5: Default fallback
    print("5. Using default fallback: (0.0, 0.0)")
    return (0.0, 0.0)

if __name__ == '__main__':
    result = debug_pricing_flow()
    print(f"\nFinal result: {result}")
    
    # Now test the actual function
    print(f"\nActual function result: {COST_TRACKER.get_model_pricing('qwen3:14b')}") 