"""Fetch a checksum-pinned CC0 clip and reproduce the resized input.

Only downloads the documented public source. Requires FFmpeg for encoding.
"""
import hashlib
from pathlib import Path
import subprocess
import urllib.request

BASE = Path(__file__).resolve().parents[1] / 'examples/natural_motion'
URL = 'https://upload.wikimedia.org/wikipedia/commons/2/2a/Big_City_Life.webm'
SHA256 = '2834fbc411cc59f06a63f9ac7ce9ba5b748b67242d1b36a4b18043a070a7966a'
BASE.mkdir(parents=True, exist_ok=True)
source = BASE / 'source.webm'
if not source.exists():
    request = urllib.request.Request(URL, headers={'User-Agent': 'SceneTrace educational example downloader'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(10_000_001)
    if len(data) > 10_000_000 or hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('Source size/checksum mismatch; no file was saved.')
    source.write_bytes(data)
if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
    raise SystemExit('Existing source checksum mismatch.')
output = BASE / 'input.mp4'
if output.exists():
    print('Verified original source. input.mp4 already exists; leaving it unchanged.')
else:
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
                    '-vf', 'scale=960:-2', '-an', '-c:v', 'libx264', '-crf', '20', str(output)], check=True)
    print('Verified original source and generated input.mp4.')
