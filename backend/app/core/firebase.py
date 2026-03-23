import json
import logging
import firebase_admin
from firebase_admin import credentials, auth
import os
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

firebase_app = None
firebase_initialized = False

# Ensure backend/.env is loaded for os.getenv() consumers in this module.
BACKEND_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(BACKEND_ENV_PATH, override=False)


def _normalize_env_json(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    return value


def _build_creds_from_discrete_env() -> Optional[dict]:
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    private_key = os.getenv("FIREBASE_PRIVATE_KEY")
    client_email = os.getenv("FIREBASE_CLIENT_EMAIL")

    if not (project_id and private_key and client_email):
        return None

    return {
        "type": "service_account",
        "project_id": project_id,
        "private_key": private_key.replace("\\n", "\n"),
        "client_email": client_email,
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""),
        "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv(
            "FIREBASE_AUTH_PROVIDER_X509_CERT_URL",
            "https://www.googleapis.com/oauth2/v1/certs",
        ),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL", ""),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN", "googleapis.com"),
    }


def _resolve_credentials_payload() -> Optional[dict]:
    # Preferred: one-line JSON in FIREBASE_CREDENTIALS_JSON
    raw_json = _normalize_env_json(os.getenv("FIREBASE_CREDENTIALS_JSON"))
    if raw_json:
        return json.loads(raw_json)

    # Backward compatible: FIREBASE_CREDENTIALS
    raw_legacy = _normalize_env_json(os.getenv("FIREBASE_CREDENTIALS"))
    if raw_legacy:
        return json.loads(raw_legacy)

    # Fallback: discrete env variables
    return _build_creds_from_discrete_env()


def _export_derived_env(cred_dict: dict) -> None:
    # Convenience env vars derived from single JSON blob
    mappings = {
        "FIREBASE_PROJECT_ID": cred_dict.get("project_id"),
        "FIREBASE_CLIENT_EMAIL": cred_dict.get("client_email"),
        "FIREBASE_PRIVATE_KEY_ID": cred_dict.get("private_key_id"),
        "FIREBASE_CLIENT_ID": cred_dict.get("client_id"),
        "FIREBASE_AUTH_URI": cred_dict.get("auth_uri"),
        "FIREBASE_TOKEN_URI": cred_dict.get("token_uri"),
        "FIREBASE_AUTH_PROVIDER_X509_CERT_URL": cred_dict.get("auth_provider_x509_cert_url"),
        "FIREBASE_CLIENT_X509_CERT_URL": cred_dict.get("client_x509_cert_url"),
        "FIREBASE_UNIVERSE_DOMAIN": cred_dict.get("universe_domain"),
    }
    for key, value in mappings.items():
        if value and not os.getenv(key):
            os.environ[key] = str(value)

    private_key = cred_dict.get("private_key")
    if private_key and not os.getenv("FIREBASE_PRIVATE_KEY"):
        os.environ["FIREBASE_PRIVATE_KEY"] = str(private_key)

def initialize_firebase():
    """Initialize Firebase Admin SDK. Safe to call multiple times."""
    global firebase_app, firebase_initialized
    
    if firebase_initialized:
        return
    
    if not (os.getenv("FIREBASE_CREDENTIALS_JSON") or os.getenv("FIREBASE_CREDENTIALS") or os.getenv("FIREBASE_PROJECT_ID")):
        logger.error("Firebase credentials env not set. Use FIREBASE_CREDENTIALS_JSON (single-line JSON).")
        firebase_initialized = True
        return
    
    try:
        cred_dict = _resolve_credentials_payload()
        if not cred_dict:
            logger.error("Firebase credentials are missing or incomplete.")
            firebase_initialized = True
            return

        _export_derived_env(cred_dict)
        cred = credentials.Certificate(cred_dict)
        firebase_app = firebase_admin.initialize_app(cred)
        firebase_initialized = True
        logger.info("Firebase Admin SDK initialized successfully")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Firebase credentials JSON: {e}")
        firebase_initialized = True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        firebase_initialized = True

def verify_token(id_token: str):
    """Verify Firebase ID token and return decoded claims."""
    if not firebase_initialized:
        initialize_firebase()
    
    if not firebase_app or not firebase_initialized:
        raise Exception("Firebase not initialized. Set FIREBASE_CREDENTIALS_JSON env var.")
    
    try:
        return auth.verify_id_token(id_token)
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise
