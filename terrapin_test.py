#!/usr/bin/env python3
import socket
import ssl
import sys
import re

def test_terrapin(hostname, port=443):
    context = ssl.create_default_context()
    
    print(f"Testing {hostname}:{port} for Terrapin vulnerability...")
    
    try:
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # Get cipher info
                cipher = ssock.cipher()
                print(f"Connected using: {cipher}")
                
                # Check if using CBC cipher (vulnerable to Terrapin)
                if "CBC" in cipher[0]:
                    print("WARNING: Server is using CBC cipher which may be vulnerable to Terrapin attacks!")
                    return True
                else:
                    print("Server is not using CBC ciphers, likely not vulnerable to Terrapin.")
                    return False
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    hostname = "aliasrobotics.com"
    test_terrapin(hostname)
