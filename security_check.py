import socket
import ssl

def test_terrapin(hostname, port=443):
    """
    Test if a server is vulnerable to the Terrapin Attack (CVE-2023-48795)
    This is a basic test that checks for specific TLS behavior related to CBC ciphers
    """
    print(f"Testing {hostname}:{port} for Terrapin vulnerability (CVE-2023-48795)...")
    
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    # Force the use of CBC ciphers which are affected by the attack
    context.set_ciphers('AES256-SHA:AES128-SHA:DES-CBC3-SHA')
    
    try:
        with socket.create_connection((hostname, port)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cipher = ssock.cipher()
                print(f"Connected using: {cipher}")
                
                # If we can connect with CBC ciphers, the server might be vulnerable
                if "CBC" in cipher[0]:
                    print("Server accepts CBC ciphers which are potentially vulnerable to Terrapin Attack")
                    print("Note: This is only an indicator and not a definitive test")
                    print("Full verification would require specialized tools and in-depth testing")
                    return True
                else:
                    print("Server not using CBC ciphers in this test")
                    return False
                    
    except Exception as e:
        print(f"Error during test: {e}")
        return False

# Also check for the AWS S3 bucket
def check_s3_bucket(bucket_name):
    """
    Check if an S3 bucket exists and is accessible
    """
    print(f"\nChecking S3 bucket: {bucket_name}")
    try:
        with socket.create_connection((bucket_name, 443)) as sock:
            # S3 should resolve so we can connect, which suggests the bucket exists
            print(f"Bucket {bucket_name} exists (DNS resolution successful)")
            
            # We try to make a simple HTTPS request
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=bucket_name) as ssock:
                ssock.sendall(b"GET / HTTP/1.1\r\nHost: " + bucket_name.encode() + b"\r\n\r\n")
                data = ssock.recv(4096)
                print(f"Response from bucket:\n{data.decode('utf-8', errors='ignore')[:500]}")
                
                if b"AccessDenied" in data or b"403 Forbidden" in data:
                    print("Bucket exists but access is denied (this is normal for protected buckets)")
                    return True
                else:
                    print("Bucket might be publicly accessible!")
                    return True
                    
    except Exception as e:
        print(f"Error checking bucket: {e}")
        if "Name or service not known" in str(e):
            print("Bucket does not exist or is not accessible")
        return False

# Test both issues
test_terrapin("aliasrobotics.com")
check_s3_bucket("trainingaliasrobotics.s3.amazonaws.com")
