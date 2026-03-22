# Firebase Setup & Deployment Guide

## Quick Start: Firebase Authentication Setup

### 1. Create a Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click "Create Project"
3. Enter project name (e.g., "sigmacloud-prod")
4. Select your region
5. Click "Create Project"

### 2. Configure Authentication
1. In Firebase Console, go to **Build > Authentication**
2. Click **Get Started**
3. Enable these providers:
   - **Email/Password**: Click "Email/Password" provider, enable both options, save
   - **Google**: Click "Google" provider, select your support email, save

### 3. Get Configuration Keys
1. Go to **Project Settings** (gear icon)
2. Click **Your apps** section
3. If no web app exists, click **Create Web App**
4. App name: "sigmacloud-web"
5. Firebase will display your config - copy these values

### 4. Configure Environment Variables

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_auth_domain
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_storage_bucket
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
NEXT_PUBLIC_API_URL=http://your-backend.com
```

### 5. Install Dependencies
```bash
cd frontend
npm install
```

### 6. Run Locally
```bash
npm run dev
```

Visit `http://localhost:3000` - you'll be redirected to login if not authenticated.

---

## Deployment Options

### Vercel (Recommended for Frontend)

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourname/sigmacloud.git
   git push -u origin main
   ```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Select your GitHub repository
   - Project name: "sigmacloud"
   - Root Directory: "frontend"

3. **Add Environment Variables in Vercel**
   - Go to Project Settings > Environment Variables
   - Add all NEXT_PUBLIC_* variables
   - Add NEXT_PUBLIC_API_URL pointing to your backend

4. **Deploy**
   - Click "Deploy"
   - Visit your Vercel URL

### Docker Deployment (Backend & Frontend)

**Build Backend Docker Image:**
```bash
cd backend
docker build -t sigmacloud-backend:latest .
```

**Build Frontend Docker Image:**
```bash
cd frontend
docker build -t sigmacloud-frontend:latest .
```

**Run with Docker Compose:**
```bash
docker-compose up -d
```

**Push to Docker Registry (e.g., Docker Hub):**
```bash
docker tag sigmacloud-backend:latest yourname/sigmacloud-backend:latest
docker push yourname/sigmacloud-backend:latest
```

### Cloud Deployment (GCP, AWS, Azure)

**Google Cloud Run (Backend):**
```bash
cd backend
gcloud run deploy sigmacloud-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**AWS Lambda (Backend):**
- Use Zappa or AWS SAM
- Configure RDS for database
- Set environment variables in Lambda console

**Azure App Service:**
- Create Web App on Azure Portal
- Connect GitHub repository
- Configure environment variables
- Enable CORS for frontend

---

## Environment Variables Reference

### Firebase Keys (from Firebase Console)
- Get from: Project Settings > Your apps > Web app config
- All keys are public (prefixed with NEXT_PUBLIC_)
- Safe to include in frontend code

### Backend Configuration
```bash
# Database (SQLite default, change for production)
DATABASE_URL=sqlite:///./sigmacloud.db

# ML Storage
MODEL_STORAGE_PATH=./storage/models
DATASET_STORAGE_PATH=./storage/datasets
```

---

## User Flow After Deployment

1. **User visits app** → Redirected to `/login`
2. **Sign up** → Firebase creates account
3. **Sign in** → Firebase authenticates
4. **Dashboard** → Protected route, only accessible when logged in
5. **Logout** → User token cleared, redirected to login

---

## Production Checklist

- [ ] Firebase project created and configured
- [ ] Authentication providers enabled (Email, Google)
- [ ] Environment variables set in deployment platform
- [ ] CORS configured on backend
- [ ] Database backed up and secured
- [ ] API keys rotated and secured
- [ ] Frontend built and deployed
- [ ] Backend deployed with proper error handling
- [ ] SSL/HTTPS enabled
- [ ] Rate limiting configured
- [ ] Monitoring and logging set up

---

## Troubleshooting

**"Cannot read properties of undefined (reading 'auth')"**
- Check NEXT_PUBLIC_FIREBASE_* variables are set
- Restart dev server after changing .env.local
- Check Firebase project ID is correct

**"FirebaseError: Missing or insufficient permissions"**
- Enable Email/Password and Google providers in Firebase Authentication
- Check Firestore/Realtime Database rules (if using)

**"401 Unauthorized on API calls"**
- Ensure backend is running
- Check CORS configuration
- Verify API_URL environment variable

**User redirected to login after page refresh**
- Normal behavior - app revalidates auth on load
- Firebase session persisted in localStorage automatically

---

## Security Notes

✅ Do:
- Use environment variables for sensitive data
- Enable reCAPTCHA on Firebase login (optional)
- Require strong passwords
- Implement rate limiting on backend
- Use HTTPS in production

❌ Don't:
- Commit `.env.local` to version control
- Expose backend API URLs publicly
- Use weak Firebase security rules
- Store tokens in localStorage for sensitive operations

---

## Support

For issues, refer to:
- [Firebase Docs](https://firebase.google.com/docs)
- [Next.js Docs](https://nextjs.org/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com)
