#!/usr/bin/env python3
import requests
import sys
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

def check_s3_bucket(bucket_name):
    """Check if an S3 bucket is publicly accessible and list its contents if possible."""
    # Remove any s3.amazonaws.com suffix if present
    if '.' in bucket_name:
        parsed = urlparse('http://' + bucket_name)
        bucket_name = parsed.netloc.split('.')[0]
    
    urls = [
        f"https://{bucket_name}.s3.amazonaws.com",
        f"https://s3.amazonaws.com/{bucket_name}"
    ]
    
    for url in urls:
        print(f"Checking {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                print("Bucket is publicly accessible!")
                
                # Try to parse XML response to list bucket contents
                try:
                    root = ET.fromstring(response.content)
                    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                    
                    print("\nContents:")
                    for content in root.findall('.//s3:Contents', ns):
                        key = content.find('s3:Key', ns)
                        size = content.find('s3:Size', ns)
                        last_modified = content.find('s3:LastModified', ns)
                        
                        if key is not None:
                            print(f"File: {key.text}", end=" ")
                            if size is not None:
                                print(f"(Size: {size.text} bytes)", end=" ")
                            if last_modified is not None:
                                print(f"Last Modified: {last_modified.text}", end="")
                            print()
                except Exception as e:
                    print(f"Error parsing bucket listing: {e}")
                    print("Raw response:")
                    print(response.text[:1000])  # Print first 1000 chars
            elif response.status_code == 403:
                print("Bucket exists but access is forbidden")
            elif response.status_code == 404:
                print("Bucket not found")
            else:
                print(f"Unexpected response with status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
    
    # Try alternate spelling as mentioned in the request
    alt_url = f"https://trainingaliasrobotics.s3.amazonnaws.com"
    print(f"\nChecking alternate spelling: {alt_url}")
    try:
        response = requests.get(alt_url, timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text[:1000]}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    bucket_name = "trainingaliasrobotics"
    check_s3_bucket(bucket_name)
