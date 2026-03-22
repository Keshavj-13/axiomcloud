from fastapi import Header, HTTPException
from app.core.firebase import verify_token

def get_current_user(authorization: str = Header(None)):
    """
    FastAPI dependency to extract and verify Firebase ID token from Authorization header.
    
    Usage:
        @router.post("/some-endpoint")
        def endpoint(user=Depends(get_current_user)):
            # user contains decoded Firebase claims (uid, email, etc.)
            ...
    
    Raises:
        HTTPException 401: If token is missing or invalid
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    # Remove "Bearer " prefix if present
    token = authorization.replace("Bearer ", "")

    try:
        decoded = verify_token(token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
