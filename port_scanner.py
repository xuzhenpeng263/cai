import socket
import sys
from datetime import datetime

# Define target
target = "192.168.1.1"
print(f"\nStarting scan on host: {target}")
print("-" * 50)

# Get the current time when scan started
t1 = datetime.now()

try:
    # Scan common ports
    common_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]
    
    print(f"Scanning common ports on {target}...")
    
    for port in common_ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.setdefaulttimeout(1)
        
        # Returns an error indicator
        result = s.connect_ex((target, port))
        
        if result == 0:
            try:
                service = socket.getservbyport(port)
                print(f"Port {port}: OPEN - {service}")
            except:
                print(f"Port {port}: OPEN - Unknown service")
        s.close()

    # Get the current time when scan completed
    t2 = datetime.now()
    
    # Calculate the difference of time to know how long the scan took
    total = t2 - t1
    
    # Print the information to screen
    print("-" * 50)
    print(f"Scanning completed in: {total}")

except KeyboardInterrupt:
    print("\nExiting program.")
    sys.exit()
except socket.gaierror:
    print("\nHostname could not be resolved.")
    sys.exit()
except socket.error:
    print("\nServer not responding.")
    sys.exit()
