import os

from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatClient

load_dotenv()

def create_chat_client() -> OpenAIChatClient:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not endpoint or not api_key or not deployment:
        raise RuntimeError(
            "Please configure AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME."
        )

    # This resource only supports the dateless v1 API; dated api-version values are rejected.
    base_url = endpoint.rstrip("/") + "/openai/v1/"

    return OpenAIChatClient(
        model=deployment,
        base_url=base_url,
        api_key=api_key,
    )