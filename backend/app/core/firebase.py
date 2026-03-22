import json
import firebase_admin
from firebase_admin import credentials, auth
import os

firebase_creds = os.getenv("FIREBASE_CREDENTIALS")

if firebase_creds:
    cred_dict = json.loads(firebase_creds)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
else:
    raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")

def verify_token(id_token: str):
    """Verify Firebase ID token and return decoded claims."""
    return auth.verify_id_token(id_token)
