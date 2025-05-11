#!/usr/bin/env python3
import socket
import sys
from datetime import datetime
import concurrent.futures

# Define the target
target = "192.168.1.1"
print(f"Starting scan of {target} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Function to scan a single port
def scan_port(port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex((target, port))
        if result == 0:
            try:
                service = socket.getservbyport(port)
                return f"Port {port}: OPEN - {service}"
            except:
                return f"Port {port}: OPEN - Unknown service"
        s.close()
    except:
        pass
    return None

# List of common ports to scan
common_ports = [20, 21, 22, 23, 25, 53, 80, 110, 123, 143, 443, 445, 3389, 8080, 8443]

print(f"Scanning common ports on {target}...")

# Use a thread pool to scan ports in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
    results = executor.map(scan_port, common_ports)
    
    # Print results
    for result in results:
        if result:
            print(result)

# Now do a scan of the first 1000 ports
print(f"\nScanning ports 1-1000 on {target}...")
open_ports = []

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = executor.map(scan_port, range(1, 1001))
    
    for result in results:
        if result:
            print(result)
            
print(f"Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
