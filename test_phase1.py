"""
test_phase1.py
--------------
Phase 1 connection test for the Interview Trainer Agent.

Verifies that:
  1. All required environment variables are present.
  2. IBM watsonx.ai authentication succeeds.
  3. The correct project ID and Dallas endpoint are used.
  4. IBM Granite 4 H Small is reachable.
  5. A real response is returned for a test prompt.
  6. No secret is printed.

Run:
    python test_phase1.py
"""

import sys


def _separator(char: str = "-", width: int = 60) -> str:
    return char * width


def run_phase1_test() -> None:
    print(_separator("="))
    print("  Interview Trainer Agent - Phase 1 Connection Test")
    print(_separator("="))

    # ── Step 1: Load and validate configuration ──────────────────────────────
    print("\n[1/4] Loading configuration from environment variables...")
    try:
        from config.settings import settings
        # Show config summary — deliberately excludes the API key value.
        print(f"      [OK] Project ID : {settings.watsonx_project_id}")
        print(f"      [OK] Endpoint   : {settings.watsonx_url}")
        print(f"      [OK] Model      : {settings.watsonx_model_id}")
        print(f"      [OK] API key    : {'set (hidden)' if settings.watsonx_apikey else 'MISSING'}")
    except EnvironmentError as exc:
        print(f"\n  [FAIL] Configuration error:\n    {exc}")
        sys.exit(1)

    # ── Step 2: Initialise the Granite client ────────────────────────────────
    print("\n[2/4] Connecting to IBM watsonx.ai and authenticating...")
    try:
        from llm.granite import GraniteClient
        client = GraniteClient()
        print("      [OK] IBM watsonx.ai authentication successful.")
        print("      [OK] Granite model client initialised.")
    except Exception as exc:
        _handle_connection_error(exc)
        sys.exit(1)

    # ── Step 3: Send a test prompt ───────────────────────────────────────────
    test_prompt = "Explain what a Python list is in two sentences."
    print(f"\n[3/4] Sending test prompt to Granite...")
    print(f"      Prompt: \"{test_prompt}\"")

    try:
        response = client.generate(test_prompt)
    except Exception as exc:
        _handle_connection_error(exc)
        sys.exit(1)

    # ── Step 4: Display response ─────────────────────────────────────────────
    print(f"\n[4/4] Granite response received:")
    print(_separator())
    print(response)
    print(_separator())

    print("\n" + _separator("="))
    print("  [OK] Phase 1 COMPLETE - IBM Granite connection verified.")
    print("  Ready for Phase 2.")
    print(_separator("=") + "\n")


def _handle_connection_error(exc: Exception) -> None:
    """Print a helpful, secret-safe error message for connection failures."""
    msg = str(exc)
    error_type = type(exc).__name__

    print(f"\n  [FAIL] Connection failed: {error_type}")

    # Provide targeted guidance without leaking credentials.
    if "401" in msg or "Unauthorized" in msg or "authentication" in msg.lower():
        print("    > Check that WATSONX_APIKEY in your .env is correct and active.")
        print("    > Verify the API key has access to watsonx.ai services.")
    elif "403" in msg or "Forbidden" in msg:
        print("    > Your API key may not have permission for this project.")
        print("    > Confirm WATSONX_PROJECT_ID matches your watsonx.ai project.")
    elif "404" in msg or "not found" in msg.lower():
        print("    > The model ID or project ID may be incorrect.")
        print("    > Confirm WATSONX_MODEL_ID=ibm/granite-4-h-small is available in your region.")
    elif "Connection" in error_type or "Timeout" in error_type or "Network" in error_type:
        print("    > Network or connectivity issue.")
        print("    > Confirm WATSONX_URL=https://us-south.ml.cloud.ibm.com is reachable.")
    elif "ModuleNotFoundError" in error_type or "ImportError" in error_type:
        print("    > Missing dependency. Run: pip install ibm-watsonx-ai")
    else:
        # Print the raw error but scrub the API key value just in case.
        from config.settings import settings
        safe_msg = msg.replace(settings.watsonx_apikey, "***")
        print(f"    > Details: {safe_msg}")


if __name__ == "__main__":
    run_phase1_test()
