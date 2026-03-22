import json
import logging
import firebase_admin
from firebase_admin import credentials, auth
import os

logger = logging.getLogger(__name__)

firebase_app = None
firebase_initialized = False

def initialize_firebase():
    """Initialize Firebase Admin SDK. Safe to call multiple times."""
    global firebase_app, firebase_initialized
    
    if firebase_initialized:
        return
    
    firebase_creds = os.getenv("FIREBASE_CREDENTIALS")
    
    if not firebase_creds:
        logger.error("FIREBASE_CREDENTIALS environment variable not set. Firebase auth will be disabled.")
        firebase_initialized = True
        return
    
    try:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_app = firebase_admin.initialize_app(cred)
        firebase_initialized = True
        logger.info("Firebase Admin SDK initialized successfully")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse FIREBASE_CREDENTIALS JSON: {e}")
        firebase_initialized = True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        firebase_initialized = True

def verify_token(id_token: str):
    """Verify Firebase ID token and return decoded claims."""
    if not firebase_initialized:
        initialize_firebase()
    
    if not firebase_app or not firebase_initialized or not os.getenv("FIREBASE_CREDENTIALS"):
        raise Exception("Firebase not initialized. Set FIREBASE_CREDENTIALS env var.")
    
    try:
        return auth.verify_id_token(id_token)
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise
