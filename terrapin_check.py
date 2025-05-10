#!/usr/bin/env python3
import socket
import ssl
import sys

def check_terrapin(hostname, port=443):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cipher = ssock.cipher()
                print(f"Connected to {hostname}:{port}")
                print(f"Cipher: {cipher}")
                
                # Check for server features that might indicate Terrapin vulnerability
                # Note: This is a simplified check and not a complete test
                if cipher[0].startswith(('TLS_RSA_', 'RSA-')):
                    print(f"[!] Potentially vulnerable to Terrapin: {hostname} uses RSA key exchange")
                else:
                    print(f"[+] Not likely vulnerable to Terrapin: {hostname} doesn't use RSA key exchange")
                
    except Exception as e:
        print(f"Error connecting to {hostname}:{port} - {str(e)}")

if __name__ == "__main__":
    check_terrapin("aliasrobotics.com")
