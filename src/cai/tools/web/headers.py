"""
Module for analyzing HTTP requests and responses with security focus.

This module provides utilities for making HTTP requests and analyzing
the responses from a security testing perspective, including header
analysis, parameter inspection, and security vulnerability detection.
"""

from urllib.parse import urlparse
from typing import Optional
import requests  # pylint: disable=E0401
from cai.sdk.agents import function_tool


@function_tool
def web_request_framework(  # noqa: E501 # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
                            url: str = "",
                            method: str = "GET",
                            headers: Optional[str] = None,
                            data: Optional[str] = None,
                            cookies: Optional[str] = None,
                            params: Optional[str] = None) -> str:
    """
    Analyze HTTP requests and responses in detail for security testing.

    Args:
        url: Target URL to analyze
        method: HTTP method (GET, POST, etc.)
        headers: Request headers
        data: Request body data
        cookies: Request cookies
        params: URL parameters
        ctf: CTF object to use for context
    Returns:
        str: Detailed analysis of the HTTP interaction including:
            - Request details (method, headers, parameters)
            - Response analysis (status, headers, body)
            - Security observations
            - Potential vulnerabilities
            - Suggested attack vectors
    """
    try:
        import json
        
        # Parse string parameters to dictionaries
        def parse_param(param_str):
            if not param_str:
                return None
            try:
                return json.loads(param_str)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, try to parse as key=value pairs
                if '=' in str(param_str):
                    result = {}
                    for pair in str(param_str).split('&'):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            result[key.strip()] = value.strip()
                    return result
                return None
        
        # Convert string parameters to dictionaries
        headers_dict = parse_param(headers)
        cookies_dict = parse_param(cookies) 
        params_dict = parse_param(params)
        data_dict = parse_param(data)
        
        # Initialize analysis results
        analysis = []
        analysis.append("\n=== HTTP Request Analysis ===\n")

        # Analyze URL structure
        parsed_url = urlparse(url)
        analysis.append("URL Analysis:")
        analysis.append(f"- Scheme: {parsed_url.scheme}")
        analysis.append(f"- Domain: {parsed_url.netloc}")
        analysis.append(f"- Path: {parsed_url.path}")
        analysis.append(f"- Parameters: {parsed_url.query}")

        # Analyze request components
        analysis.append("\nRequest Details:")
        analysis.append(f"- Method: {method}")

        if headers_dict:
            analysis.append("\nHeaders Analysis:")
            for header, value in headers_dict.items():
                analysis.append(f"- {header}: {value}")

        if cookies_dict:
            analysis.append("\nCookies Analysis:")
            for cookie, value in cookies_dict.items():
                analysis.append(f"- {cookie}: {value}")

        if params_dict:
            analysis.append("\nParameters Analysis:")
            for param, value in params_dict.items():
                analysis.append(f"- {param}: {value}")

        if data_dict:
            analysis.append("\nBody Data Analysis:")
            for key, value in data_dict.items():
                analysis.append(f"- {key}: {value}")

        # Make the request and analyze response
        response = requests.request(
            method=method,
            url=url,
            headers=headers_dict,
            data=data_dict,
            cookies=cookies_dict,
            params=params_dict,
            verify=False,
            allow_redirects=True
        )

        analysis.append("\n=== HTTP Response Analysis ===\n")
        analysis.append(f"Status Code: {response.status_code}")

        analysis.append("\nResponse Headers:")
        for header, value in response.headers.items():
            analysis.append(f"- {header}: {value}")

        analysis.append(f"\nResponse Size: {len(response.content)} bytes")

        # Security observations
        analysis.append("\n=== Security Analysis ===\n")

        # Check security headers
        security_headers = [
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'X-Frame-Options',
            'X-XSS-Protection',
            'X-Content-Type-Options'
        ]

        missing_headers = []
        for header in security_headers:
            if header not in response.headers:
                missing_headers.append(header)

        if missing_headers:
            analysis.append("Missing Security Headers:")
            for header in missing_headers:
                analysis.append(f"- {header}")

        # Check for sensitive information
        sensitive_patterns = [
            'password',
            'token',
            'key',
            'secret',
            'admin',
            'root'
        ]

        for pattern in sensitive_patterns:
            if pattern in response.text.lower():
                analysis.append(
                    f"\nPotential sensitive information found: '{pattern}'")

        return "\n".join(analysis)

    except Exception as e:  # pylint: disable=broad-except
        return f"Error analyzing request: {str(e)}"
