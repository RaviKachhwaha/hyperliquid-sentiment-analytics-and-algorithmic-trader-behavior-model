import os
import sys
import urllib.request
import re

def download_gdrive_file(file_id, destination):
    print(f"Starting download for file ID: {file_id} to {destination}")
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    # Try using requests if available
    try:
        import requests
        session = requests.Session()
        response = session.get(url, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            print(f"Found download warning confirmation token, retrying with token...")
            response = session.get(url, params={'id': file_id, 'confirm': token}, stream=True)
            
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
        return True
    except ImportError:
        pass
    
    # Fallback using urllib
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read(1024 * 1024)
            html = content.decode('utf-8', errors='ignore')
            confirm_match = re.search(r'confirm=([A-Za-z0-9_]+)', html)
            
            if confirm_match:
                confirm_token = confirm_match.group(1)
                print(f"Found warning token: {confirm_token}")
                confirm_url = f"https://docs.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                req = urllib.request.Request(confirm_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as confirm_response:
                    with open(destination, 'wb') as f:
                        f.seek(0)
                        while True:
                            chunk = confirm_response.read(1024*1024)
                            if not chunk:
                                break
                            f.write(chunk)
            else:
                with open(destination, 'wb') as f:
                    f.write(content)
                    while True:
                        chunk = response.read(1024*1024)
                        if not chunk:
                            break
                        f.write(chunk)
        print("Download complete using fallback.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Files to download
    files = {
        'bitcoin_sentiment.csv': '1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf',
        'historical_trader_data.csv': '1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs'
    }
    
    for filename, file_id in files.items():
        dest = os.path.join('data', filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"{filename} already exists and is non-empty, skipping download.")
        else:
            success = download_gdrive_file(file_id, dest)
            if not success:
                print(f"Failed to download {filename}")
                sys.exit(1)
    print("All downloads finished.")
