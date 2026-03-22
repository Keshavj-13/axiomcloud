"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";
import { auth, firebaseConfigured } from "@/lib/firebase";
import toast from "react-hot-toast";
import { Mail, Lock, LogIn, UserPlus, Chrome } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSignup, setIsSignup] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!firebaseConfigured || !auth) {
      toast.error("Firebase keys are missing. Add NEXT_PUBLIC_FIREBASE_* variables.");
      return;
    }
    if (!email || !password) {
      toast.error("Please enter email and password");
      return;
    }

    setLoading(true);
    try {
      if (isSignup) {
        await createUserWithEmailAndPassword(auth, email, password);
        toast.success("Account created! Redirecting...");
      } else {
        await signInWithEmailAndPassword(auth, email, password);
        toast.success("Logged in! Redirecting...");
      }
      router.push("/dashboard");
    } catch (error: any) {
      const msg = error.code === "auth/email-already-in-use"
        ? "Email already in use"
        : error.code === "auth/weak-password"
        ? "Password should be at least 6 characters"
        : error.code === "auth/invalid-email"
        ? "Invalid email address"
        : error.code === "auth/user-not-found"
        ? "User not found"
        : error.code === "auth/wrong-password"
        ? "Incorrect password"
        : error.message || "Authentication failed";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    if (!firebaseConfigured || !auth) {
      toast.error("Firebase keys are missing. Add NEXT_PUBLIC_FIREBASE_* variables.");
      return;
    }
    setLoading(true);
    try {
      const provider = new GoogleAuthProvider();
      await signInWithPopup(auth, provider);
      toast.success("Signed in with Google! Redirecting...");
      router.push("/dashboard");
    } catch (error: any) {
      toast.error(error.message || "Google sign-in failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-600 via-primary-500 to-primary-400 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-surface rounded-2xl shadow-2xl p-8 space-y-8">
          {/* Header */}
          <div className="text-center">
            <h1 className="text-3xl font-bold text-text-primary mb-2">
              Axiom Cloud
            </h1>
            <p className="text-text-secondary">
              {isSignup ? "Create your account" : "Welcome back"}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleEmailAuth} className="space-y-4">
            {!firebaseConfigured && (
              <div className="rounded-lg border border-amber-300/50 bg-amber-100/50 px-3 py-2 text-xs text-amber-900">
                Firebase config not found. Add NEXT_PUBLIC_FIREBASE_* keys in frontend/.env.local.
              </div>
            )}
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 w-5 h-5 text-outline" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full pl-10 pr-4 py-2 border border-outline rounded-lg bg-surface text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500"
                  disabled={loading || !firebaseConfigured}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 w-5 h-5 text-outline" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-4 py-2 border border-outline rounded-lg bg-surface text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500"
                  disabled={loading || !firebaseConfigured}
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || !firebaseConfigured}
              className="w-full bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : isSignup ? (
                <>
                  <UserPlus className="w-5 h-5" />
                  Sign Up
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  Sign In
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-outline" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-surface text-text-secondary">
                or continue with
              </span>
            </div>
          </div>

          {/* Google Button */}
          <button
            type="button"
            onClick={handleGoogleSignIn}
            disabled={loading || !firebaseConfigured}
            className="w-full border border-outline hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed text-text-primary font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2"
          >
            <Chrome className="w-5 h-5" />
            Google
          </button>

          {/* Toggle */}
          <div className="text-center text-text-secondary">
            {isSignup ? (
              <>
                Already have an account?{" "}
                <button
                  onClick={() => setIsSignup(false)}
                  className="text-primary-600 hover:text-primary-700 font-semibold"
                >
                  Sign In
                </button>
              </>
            ) : (
              <>
                Don't have an account?{" "}
                <button
                  onClick={() => setIsSignup(true)}
                  className="text-primary-600 hover:text-primary-700 font-semibold"
                >
                  Sign Up
                </button>
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-white/80 text-sm">
          <p>© 2026 Axiom Cloud. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}
