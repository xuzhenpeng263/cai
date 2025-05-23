#!/usr/bin/env python3

import requests
import json

def debug_litellm_pricing():
    """Debug why LiteLLM returns pricing for local models"""
    
    LITELLM_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
    
    try:
        response = requests.get(LITELLM_URL, timeout=5)
        data = response.json()
        
        print('Checking if qwen3:14b exists in LiteLLM pricing...')
        if 'qwen3:14b' in data:
            print('Found qwen3:14b in LiteLLM:', data['qwen3:14b'])
        else:
            print('qwen3:14b NOT found in LiteLLM pricing')
            
        print('\nChecking similar qwen models...')
        qwen_models = [k for k in data.keys() if 'qwen' in k.lower()]
        print(f'Total Qwen models found in LiteLLM: {len(qwen_models)}')
        print('First 10 Qwen models:')
        for model in sorted(qwen_models)[:10]:
            print(f'  {model}: {data[model]}')
            
        # Check if there's a pattern match or fallback
        print('\nChecking for pattern matches...')
        potential_matches = [k for k in data.keys() if 'qwen' in k.lower() and '14b' in k.lower()]
        if potential_matches:
            print('Models with qwen and 14b:')
            for model in potential_matches:
                print(f'  {model}: {data[model]}')
        
        # Test our current logic
        print('\nTesting our get_model_pricing logic...')
        from src.cai.util import COST_TRACKER
        pricing = COST_TRACKER.get_model_pricing('qwen3:14b')
        print(f'Our function returns for qwen3:14b: {pricing}')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    debug_litellm_pricing() 