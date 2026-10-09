"""GenEval2 Soft-TIFA reward client.

The reward itself is computed by ``scripts/geneval2_server.py``, an HTTP server
that runs Qwen3-VL-2B-Instruct as the VQA model (it needs a newer
``transformers`` than the SD3 training environment, so it runs as a separate
process). This module sends rendered images plus their GenEval2 metadata
(``vqa_list``) to that server.

Environment variables:
    GENEVAL2_SERVER_URLS     comma-separated server URLs; rank r uses URL r % len
    GENEVAL2_SERVER_URL      single server URL (default http://127.0.0.1:18185)
    GENEVAL2_METHOD          soft_tifa_gm (default, used in the paper) or soft_tifa_am
    GENEVAL2_BATCH_SIZE      images per request (default 16)
    GENEVAL2_REQUEST_TIMEOUT seconds (default 900)
"""

import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image


def geneval2_score(device):
    """Submit images to GenEval2 Soft-TIFA reward servers.

    Metadata must come from the GenEval2 jsonl rows and include `vqa_list`.
    The selected training score is controlled by GENEVAL2_METHOD:
    `soft_tifa_gm` by default, or `soft_tifa_am`.
    """
    import pickle
    import requests
    from io import BytesIO
    from requests.adapters import HTTPAdapter, Retry

    batch_size = int(os.environ.get("GENEVAL2_BATCH_SIZE", "16"))
    request_timeout = float(os.environ.get("GENEVAL2_REQUEST_TIMEOUT", "900"))
    method = os.environ.get("GENEVAL2_METHOD", "soft_tifa_gm")
    server_urls_env = os.environ.get("GENEVAL2_SERVER_URLS", "").strip()
    if server_urls_env:
        server_urls = [u.strip().rstrip("/") for u in server_urls_env.split(",") if u.strip()]
        if not server_urls:
            raise ValueError("GENEVAL2_SERVER_URLS was set but no valid URLs were parsed")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", str(local_rank)))
        url = server_urls[global_rank % len(server_urls)]
        print(
            f"[GenEval2] Rank {global_rank} local_rank {local_rank} using reward server {url}",
            flush=True,
        )
    else:
        url = os.environ.get("GENEVAL2_SERVER_URL", "http://127.0.0.1:18185")

    sess = requests.Session()
    retries = Retry(
        total=1000,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=False,
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))

        all_scores = []
        details = defaultdict(list)
        for image_batch, metadata_batch in zip(images_batched, metadatas_batched):
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())
            payload = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batch),
                "method": method,
            }
            response = sess.post(url, data=pickle.dumps(payload), timeout=request_timeout)
            response.raise_for_status()
            response_data = pickle.loads(response.content)
            if "error" in response_data:
                raise RuntimeError(f"GenEval2 server {url} failed: {response_data['error']}")
            all_scores.extend(response_data["scores"])
            for key in ("geneval2_am", "geneval2_gm", "geneval2_question_count", "geneval2_atom_count"):
                details[key].extend(response_data.get(key, []))
        return all_scores, dict(details)

    return _fn


def multi_score(device, score_dict):
    """Weighted sum of the configured rewards. Only ``geneval2`` is provided."""
    score_functions = {
        "geneval2": geneval2_score,
    }
    score_fns = {}
    for score_name in score_dict:
        if score_name not in score_functions:
            raise ValueError(f"Unknown reward {score_name!r}; available: {sorted(score_functions)}")
        score_fns[score_name] = score_functions[score_name](device)

    def _fn(images, prompts, metadata):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            for key, value in rewards.items():
                score_details[key] = value
            score_details[score_name] = scores
            if float(weight) == 0.0:
                weighted_scores = [0.0 for _ in scores]
            else:
                weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn
