## STEP 3-7: Firebase Authentication Testing Guide

### What's Been Implemented

✅ **backend/app/core/firebase.py** - Firebase Admin SDK initialization
✅ **backend/app/api/deps.py** - Auth dependency for protecting endpoints
✅ **backend/app/api/training.py** - `/api/train-model` endpoint now requires auth
✅ **frontend/src/lib/api.ts** - All API calls automatically attach Firebase token

---

## Testing Procedure

### BEFORE TESTING

**Ensure you have set FIREBASE_CREDENTIALS on the backend:**

```bash
export FIREBASE_CREDENTIALS='{"type":"service_account","project_id":"...","...":"..."}'
```

Without this, the backend will crash on startup. This is expected—it ensures you can't accidentally deploy without Firebase.

---

### TEST CASE 1: Without Login (Should Return 401)

**Scenario:** Call `/api/train-model` without authentication token

#### Option A: Using cURL (without token)
```bash
curl -X POST https://axiomcloud.onrender.com/api/train-model \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "target_column": "target",
    "task_type": "classification"
  }'
```

**Expected Response:**
```json
{
  "detail": "Missing authorization token"
}
```

**Status:** `401 Unauthorized`

---

### TEST CASE 2: With Login (Should Succeed)

**Scenario:** Sign into the app, then trigger a training job from the dashboard

#### Step 1: Frontend - Login
1. Go to `http://localhost:3001/login` (or `https://axiomcloud.vercel.app/login` if deployed)
2. Sign up / Log in with Firebase credentials
3. You'll be redirected to `/dashboard`

#### Step 2: Frontend - Trigger Training
1. On the dashboard, select a dataset
2. Click "Start Training" or similar
3. The `trainingAPI.train()` call will automatically:
   - Get the current user's Firebase ID token via `getAuth().currentUser.getIdToken()`
   - Attach it to the request: `Authorization: Bearer <token>`
   - Send to backend

#### Step 3: Backend - Validate Token
1. Backend receives the request in `/api/train-model`
2. The `get_current_user` dependency:
   - Extracts the token from the `Authorization` header
   - Calls `verify_token(token)` from Firebase Admin SDK
   - If valid, returns the decoded claims (uid, email, etc.)
   - If invalid/expired, raises `HTTPException(401, "Invalid token")`

#### Step 4: Expected Success Response
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "in_progress",
  "dataset_id": 1,
  "created_at": "2026-03-22T10:15:00Z",
  "...": "..."
}
```

**Status:** `200 OK`

---

## How It Works End-to-End

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Logs In (Frontend)                                  │
│    Login page → Firebase Auth → ID token stored in memory   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. User Triggers Training (Frontend)                        │
│    Dashboard → trainingAPI.train(config)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Axios Request Interceptor (api.ts)                       │
│    • getIdToken() from current user                         │
│    • Attach to request: Authorization: Bearer <token>       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Request Sent to Backend (HTTPS)                          │
│    POST /api/train-model                                    │
│    Headers: Authorization: Bearer <token>                   │
│    Body: { dataset_id, target_column, ... }                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Backend Auth Dependency (deps.py)                        │
│    • Extract "Bearer <token>" from header                   │
│    • Call Firebase Admin: verify_id_token(token)            │
│    • If valid → return decoded claims (uid, email, etc.)    │
│    • If invalid → raise HTTPException(401)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Endpoint Execution (training.py)                         │
│    • Receive user claims as function parameter              │
│    • Execute training logic                                 │
│    • Return job_id and status                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Response Sent to Frontend                                │
│    • 200 OK with job_id                                     │
│    • Frontend stores job_id for polling progress            │
└─────────────────────────────────────────────────────────────┘
```

---

## Common Issues & Fixes

### ❌ Backend Crashes on Startup
```
ValueError: FIREBASE_CREDENTIALS environment variable is not set
```

**Fix:** Set the environment variable before starting the app
```bash
export FIREBASE_CREDENTIALS='{"type":"service_account",...}'
# Or on Render: Add secret in Dashboard → Environment
```

---

### ❌ Frontend Gets 401 "Missing authorization token"
**Cause:** User not logged in, or token extraction failed

**Fix:**
1. Ensure user is logged in: check `auth.currentUser` in browser console
2. Check browser Network tab: is `Authorization` header present in request?
3. If missing, debug `api.ts` interceptor

---

### ❌ Frontend Gets 401 "Invalid token"
**Cause:** Token is expired or Firebase Admin can't verify it

**Fix:**
1. Token expires after ~1 hour. Refresh the page to get a new one
2. Verify `FIREBASE_CREDENTIALS` contains the service account from the correct Firebase project
3. Check backend logs: `docker logs <container>`

---

### ❌ CORS Error
```
Access to XMLHttpRequest blocked by CORS policy
```

**Fix:** Backend `CORS_ORIGINS` already includes `http://localhost:3001` and Vercel domains.
If you get this, check:
1. Frontend is actually sending `Origin` header (automatic for same-site POST)
2. Backend has `CORSMiddleware` configured
3. Deployed backend URL in `.env.local` matches the actual URL

---

## How to Protect More Endpoints

Once you've verified `/api/train-model` works, protect other endpoints:

```python
# In any router file (e.g., backend/app/api/predictions.py)

from fastapi import Depends
from app.api.deps import get_current_user

@router.post("/predict")
def predict(
    data: PredictionInput,
    user=Depends(get_current_user),  # ← Add this
    db: Session = Depends(get_db)
):
    """Protected endpoint - requires Firebase auth."""
    # user contains: {'sub': uid, 'email': '...', 'name': '...', ...}
    user_id = user.get('sub')  # Firebase UID
    ...
```

---

## Rollout Checklist

- [ ] FIREBASE_CREDENTIALS set on backend (Render or local)
- [ ] `backend/app/core/firebase.py` created
- [ ] `backend/app/api/deps.py` created
- [ ] `/api/train-model` protected with `Depends(get_current_user)`
- [ ] `frontend/src/lib/api.ts` has request interceptor
- [ ] Test Case 1 passes (401 without login)
- [ ] Test Case 2 passes (200 with login)
- [ ] Additional endpoints protected (if needed)
- [ ] Deployed to Render + Vercel

---

## Next Steps

After testing passes:

1. **Protect remaining endpoints** - Apply same pattern to other sensitive routes
2. **Add user tracking** - Store `user_id` with training jobs, predictions, etc.
3. **Implement audit logging** - Log all protected endpoint calls with user ID
4. **Rate limiting** - Prevent abuse per user or IP
5. **Role-based access control** - If needed, restrict endpoints by user role
