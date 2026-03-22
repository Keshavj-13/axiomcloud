"""
User Authentication & Management
Lightweight user tracking (Firebase handles actual auth)
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.core.database import get_db
from app.core.database import Base, engine
from sqlalchemy import Column, String, DateTime, Integer

logger = logging.getLogger(__name__)
router = APIRouter()


# Create users table if it doesn't exist
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    display_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)


@router.post("/auth/user")
def register_or_update_user(
    firebase_uid: str,
    email: str,
    display_name: str = None,
    db: Session = Depends(get_db)
):
    """
    Called by frontend after successful Firebase auth.
    Creates or updates user record for tracking.
    """
    try:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if user:
            # Update last_login
            user.last_login = datetime.utcnow()
            if display_name:
                user.display_name = display_name
            db.commit()
            return {"id": user.id, "email": user.email, "status": "updated"}
        else:
            # Create new user
            user = User(
                firebase_uid=firebase_uid,
                email=email,
                display_name=display_name
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return {"id": user.id, "email": user.email, "status": "created"}
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Failed to register user")


@router.get("/auth/user/{firebase_uid}")
def get_user(firebase_uid: str, db: Session = Depends(get_db)):
    """Get user information by Firebase UID."""
    try:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "created_at": user.created_at,
            "last_login": user.last_login
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user")
