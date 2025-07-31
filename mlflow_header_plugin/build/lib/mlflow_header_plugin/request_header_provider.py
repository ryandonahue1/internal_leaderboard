import os
from mlflow.tracking.request_header.abstract_request_header_provider import (
    RequestHeaderProvider,
)


class PluginRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def in_context(self):
        return os.getenv("MLFLOW_TRACKING_URI") == "https://mlflow-tracking-api.vlex.io"

    def request_headers(self):
        return {"x-mlflow-user": os.getenv("MLFLOW_USER")}
