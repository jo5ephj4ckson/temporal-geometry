import requests

# Updated ANU QRNG URL Endpoint
ANU_URL = "https://qrng.anu.edu.au/wp-content/plugins/ANU-QRNG/QuantumLocker.php"
# Alternative modern endpoint if the above redirects:
# ANU_URL = "https://qrng.anu.edu.au/API/backend-getdata.php?type=hex16&length=1&size=1"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_quantum_byte():
    """Fetches raw quantum entropy from ANU with updated headers and fallback endpoints."""
    # Target 1: ANU Quantum Endpoint (Hex stream)
    try:
        url = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint8"
        response = requests.get(url, headers=HEADERS, timeout=5)
        
        # Verify we actually got JSON before attempting to parse
        if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
            data = response.json()
            if data.get('success'):
                return data['data'][0], True
    except Exception as e:
        pass

    # Fallback Target 2: Vacuum Fluctuations via Quantum-Safe JSON API
    try:
        url = "https://qrng.anu.edu.au/wp-content/plugins/ANU-QRNG/QuantumLocker.php?contentType=json&length=1&type=uint8"
        response = requests.get(url, headers=HEADERS, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['data'][0], True
    except Exception:
        pass

    return None, False