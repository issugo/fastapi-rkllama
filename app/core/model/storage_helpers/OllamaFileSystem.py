import json
from typing import Tuple, Any
import requests

from core.model.storage_helpers.OllamaModelStorageHelper import BLOBS
from core.model.storage_helpers.OllamaStorageHelper import MANIFESTS

REGISTRY_BASE = "https://registry.ollama.ai/library"
REGISTRY_API_BASE = "https://registry.ollama.ai/v2/library"


class OllamaFileSystem:
    @staticmethod
    def model_path(model_name: str, api: bool = True):
        if not api:
            return f"{REGISTRY_BASE}/{model_name}"
        return f"{REGISTRY_API_BASE}/{model_name}"

    @staticmethod
    def manifest(model_name: str, target_tag: str):
        model_path = OllamaFileSystem.model_path(model_name)
        manifest_url = f"{model_path}/{MANIFESTS}/{target_tag}"
        manifest_headers = {
            "Accept": ",".join(
                [
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                ]
            )
        }
        man_resp = requests.get(manifest_url, headers=manifest_headers, timeout=20)
        if man_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to get manifest for {model_name}:{target_tag} - HTTP {man_resp.status_code}"
            )

        return man_resp.json()

    @staticmethod
    def blob_url(digest: str, model_name: str) -> str:
        if digest:
            model_path = OllamaFileSystem.model_path(model_name)
            blob_url = f"{model_path}/{BLOBS}/{digest}"
            return blob_url
        raise ValueError("invalid blob digest")

    @staticmethod
    def model_url(model_digest: str, model_name: str) -> str:
        if model_digest:
            return OllamaFileSystem.blob_url(model_digest, model_name)
        raise ValueError("invalid model digest")

    @staticmethod
    def load_config(
        config_digest: str, model_name: str, target_tag: str
    ) -> Tuple[Any, dict | None]:
        if config_digest:
            model_path = OllamaFileSystem.model_path(model_name)
            cfg_url = f"{model_path}/{BLOBS}/{config_digest}"
            cfg_resp = requests.get(cfg_url, timeout=20)

            if cfg_resp.status_code == 200:
                # Config may be JSON (OCI image config) or sometimes other content
                try:
                    cfg = cfg_resp.json()

                    # Common places to find description/title
                    candidates = [
                        cfg.get("description"),
                        cfg.get("config", {}).get("description"),
                        cfg.get("config", {})
                        .get("Labels", {})
                        .get("org.opencontainers.image.description"),
                        cfg.get("config", {}).get("Labels", {}).get("description"),
                        cfg.get("config", {})
                        .get("Labels", {})
                        .get("org.opencontainers.image.title"),
                        cfg.get("annotations", {}).get(
                            "org.opencontainers.image.description"
                        ),
                    ]
                    config_desc = f"{model_name}:{target_tag}"
                    for c in candidates:
                        if isinstance(c, str) and c.strip():
                            config_desc = c.strip()
                            break

                    info = {
                        "description": config_desc,
                        "architecture": cfg.get("architecture"),
                        "os": cfg.get("os"),
                        "created": cfg.get("created"),
                        "labels": (
                            cfg.get("config", {}).get("Labels", {})
                            if isinstance(cfg.get("config"), dict)
                            else {}
                        ),
                    }
                    return cfg, info
                except json.JSONDecodeError as e:
                    # Non-JSON config blob; skip description
                    raise ValueError("invalid config JSON", e)

        raise ValueError("invalid config digest")
