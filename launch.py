#!/usr/bin/env python3

from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://huggingface.co/b4rtaz/DiffusionGemma-26B-Q40-Distributed-Diffusion-LLM/resolve/main"
MODEL_PARTS = [
    f"{BASE_URL}/diffusion-gemma-q40.ddlm.part{i}?download=true"
    for i in range(4)
]
TOKENIZER = f"{BASE_URL}/diffusion-gemma.ddlt?download=true"


def download(urls, destination):
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return

    temporary = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
    with temporary.open("wb") as output:
        for url in urls:
            print(f"Downloading {url}")
            with urlopen(url) as response:
                expected = response.headers.get("Content-Length")
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    print(f"\rDownloaded {output.tell() // (1024 * 1024)} MiB", end="", flush=True)
                print()
                if expected is not None and downloaded != int(expected):
                    raise RuntimeError(
                        f"Incomplete download: expected {expected} bytes, got {downloaded}"
                    )
    temporary.replace(destination)
    print(f"Created {destination}")


models = Path(__file__).resolve().parent / "models"
models.mkdir(exist_ok=True)
download(MODEL_PARTS, models / "diffusion-gemma-q40.ddlm")
download([TOKENIZER], models / "diffusion-gemma.ddlt")
