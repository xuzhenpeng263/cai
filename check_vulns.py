import subprocess
import json
import sys
import socket
import ssl
import requests
from urllib.parse import urlparse

def check_terrapin(hostname, port=443):
    print(f"[*] Checking {hostname}:{port} for Terrapin vulnerability (CVE-2023-48618)")
    try:
        # Create a socket and wrap it with SSL
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # Check if server supports TLS 1.3
                version = ssock.version()
                print(f"[*] Server SSL/TLS version: {version}")
                
                # Terrapin affects TLS 1.3 implementation with specific configurations
                if 'TLSv1.3' in version:
                    print("[!] Server supports TLS 1.3, which may be vulnerable to Terrapin attack")
                    print("[!] Note: Detailed testing requires specialized tools, as this is a downgrade attack")
                    print("[!] The vulnerability allows attackers to force downgrade to CBC ciphers in TLS 1.2")
                else:
                    print("[*] Server does not use TLS 1.3, not directly affected by Terrapin")
    except Exception as e:
        print(f"[-] Error checking for Terrapin: {e}")

def check_s3_bucket(bucket_name):
    print(f"[*] Checking S3 bucket: {bucket_name}")
    
    # Clean up the bucket name if needed
    if bucket_name.startswith("http://") or bucket_name.startswith("https://"):
        parsed = urlparse(bucket_name)
        bucket_name = parsed.netloc
    
    if ".s3.amazonaws" in bucket_name:
        bucket_name = bucket_name.split(".s3.amazonaws")[0]
    
    # Direct URL
    url = f"http://{bucket_name}.s3.amazonaws.com"
    try:
        response = requests.get(url)
        print(f"[*] Bucket URL: {url}")
        print(f"[*] Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("[!] Bucket is publicly accessible")
            print("[*] Response content:")
            # Print the first 1000 characters of the response content
            print(response.text[:1000])
        elif response.status_code == 403:
            print("[!] Bucket exists but access is forbidden (listing denied)")
            print("[*] This could mean the bucket exists but is not publicly listable")
        elif response.status_code == 404:
            print("[*] Bucket not found")
        else:
            print(f"[*] Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"[-] Error checking bucket: {e}")

    # Try to list bucket objects with aws cli
    try:
        print("[*] Attempting to list bucket objects with AWS CLI...")
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{bucket_name}", "--no-sign-request"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout:
            print("[!] Bucket is listable without authentication!")
            print("[*] Bucket contents:")
            print(result.stdout)
        else:
            print("[*] Bucket is not listable without authentication")
            if result.stderr:
                print(f"[*] Error: {result.stderr}")
    except Exception as e:
        print(f"[-] Error using AWS CLI: {e}")

if __name__ == "__main__":
    # Check for Terrapin vulnerability
    check_terrapin("aliasrobotics.com")
    
    # Check for exposed S3 bucket
    check_s3_bucket("trainingaliasrobotics")
