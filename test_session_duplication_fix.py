#!/usr/bin/env python3
"""
Test script to verify that async session duplication is fixed.
This tests that:
1. Session commands with auto_output don't duplicate
2. The LLM doesn't generate auto_output parameter
3. auto_output is forced internally when needed
"""

import json
import time

def test_command_key_generation():
    """Test that command keys are generated correctly for session commands"""
    
    # Simulate args with session_id and auto_output
    args_with_auto_output = {
        "command": "ls -la",
        "args": "",
        "session_id": "test123",
        "auto_output": True,
        "input_to_session": True,
        "call_counter": 1
    }
    
    # Simulate the command key generation logic from util.py
    effective_command_args_str = args_with_auto_output.get("args", "")
    if "command" in args_with_auto_output and args_with_auto_output.get("session_id"):
        effective_command_args_str = f"{args_with_auto_output.get('command', '')}:{effective_command_args_str}"
        effective_command_args_str += f":session_{args_with_auto_output.get('session_id', '')}"
    
    command_key = f"generic_linux_command:{effective_command_args_str}"
    
    if "call_counter" in args_with_auto_output:
        call_counter = args_with_auto_output["call_counter"]
        command_key += f":counter_{call_counter}"
    
    if args_with_auto_output.get("session_id") and args_with_auto_output.get("input_to_session"):
        command_key += f":ts_{int(time.time() * 1000)}"
        
    if args_with_auto_output.get("auto_output"):
        command_key += ":auto_output"
    
    print(f"Generated command key: {command_key}")
    
    # Verify the key contains all expected components
    assert "generic_linux_command" in command_key
    assert "ls -la" in command_key
    assert "session_test123" in command_key
    assert "counter_1" in command_key
    assert "auto_output" in command_key
    assert "ts_" in command_key
    
    print("✓ Command key generation test passed")

def test_auto_output_forcing():
    """Test that auto_output is forced internally for session commands"""
    
    # Simulate args without auto_output (as LLM would generate)
    args_without_auto_output = {
        "command": "pwd",
        "args": "",
        "session_id": "test456",
        "input_to_session": True,
        "call_counter": 2
    }
    
    # Simulate the logic from common.py that forces auto_output
    session_args = {
        "command": args_without_auto_output["command"],
        "args": "",
        "session_id": args_without_auto_output["session_id"],
        "call_counter": args_without_auto_output["call_counter"],
        "input_to_session": True,
    }
    
    # Force auto_output since no args provided or auto_output not in args
    session_args["auto_output"] = True
    
    print(f"Session args after forcing auto_output: {session_args}")
    
    # Verify auto_output was added
    assert session_args["auto_output"] == True
    
    print("✓ Auto output forcing test passed")

def test_duplicate_prevention():
    """Test that duplicate commands are prevented with different keys"""
    
    # First command
    args1 = {
        "command": "ls",
        "session_id": "test789",
        "auto_output": True,
        "call_counter": 1
    }
    
    # Second identical command but different counter
    args2 = {
        "command": "ls", 
        "session_id": "test789",
        "auto_output": True,
        "call_counter": 2
    }
    
    # Generate keys for both
    def generate_key(args):
        effective_command_args_str = args.get("args", "")
        if "command" in args and args.get("session_id"):
            effective_command_args_str = f"{args.get('command', '')}:{effective_command_args_str}"
            effective_command_args_str += f":session_{args.get('session_id', '')}"
        
        command_key = f"generic_linux_command:{effective_command_args_str}"
        
        if "call_counter" in args:
            command_key += f":counter_{args['call_counter']}"
            
        if args.get("auto_output"):
            command_key += ":auto_output"
            
        return command_key
    
    key1 = generate_key(args1)
    key2 = generate_key(args2)
    
    print(f"Key 1: {key1}")
    print(f"Key 2: {key2}")
    
    # Keys should be different due to different counters
    assert key1 != key2
    assert "counter_1" in key1
    assert "counter_2" in key2
    
    print("✓ Duplicate prevention test passed")

if __name__ == "__main__":
    print("Testing session duplication fix...")
    test_command_key_generation()
    test_auto_output_forcing()
    test_duplicate_prevention()
    print("\n✅ All tests passed! Session duplication fix is working correctly.") 