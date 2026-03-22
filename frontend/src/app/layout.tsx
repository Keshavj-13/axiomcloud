import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "react-hot-toast";
import { AuthProvider } from "@/lib/auth-context";

export const metadata: Metadata = {
  title: "Axiom Cloud AI — AutoML Platform",
  description: "Upload datasets, train models automatically, visualize results, and deploy predictions.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              style: {
                background: "#19191d",
                color: "#e7e4ec",
                border: "1px solid rgba(71,71,77,0.5)",
                fontFamily: "'Inter', sans-serif",
              },
            }}
          />
        </AuthProvider>
      </body>
    </html>
  );
}
