"""Download an explicitly pinned release artifact; never load an unchecked checkpoint."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request


def fetch(url, sha256, destination):
    destination = Path(destination)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
        return
    if not url.startswith('https://') or len(sha256) != 64:
        raise ValueError('An HTTPS URL and SHA-256 are required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError('Artifact SHA-256 mismatch')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    fetch(a.url, a.sha256, a.output)
    print('Verified artifact:', a.output)
