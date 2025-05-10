import socket
import ssl
import sys

def check_terrapin(hostname, port=443):
    """
    Basic check for Terrapin vulnerability (CVE-2023-45866)
    This is a simplified check and may not be 100% accurate
    """
    try:
        # Create socket and wrap with SSL
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Check if supports TLS 1.2 (Terrapin affects TLS 1.2 and below)
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cipher = ssock.cipher()
                if cipher and cipher[0]:
                    cipher_name = cipher[0]
                    print(f"Connected using: {cipher_name}")
                    
                    # Check if using CBC cipher (Terrapin affects CBC ciphers)
                    if 'CBC' in cipher_name:
                        print(f"[POTENTIALLY VULNERABLE] {hostname} might be vulnerable to Terrapin")
                        print(f"Using CBC cipher: {cipher_name}")
                        return True
                    else:
                        print(f"[LIKELY SAFE] {hostname} is using non-CBC cipher: {cipher_name}")
                        return False
    except ssl.SSLError as e:
        print(f"SSL Error: {e}")
    except socket.error as e:
        print(f"Socket Error: {e}")
    except Exception as e:
        print(f"Error checking {hostname}: {e}")
    
    return None

if __name__ == "__main__":
    hostname = "aliasrobotics.com"
    print(f"Checking {hostname} for potential Terrapin vulnerability...")
    check_terrapin(hostname)
