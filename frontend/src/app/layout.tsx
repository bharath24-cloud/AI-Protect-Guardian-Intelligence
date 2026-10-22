import type { Metadata } from "next";
import "./globals.css";
import LabLayout from "@/components/LabLayout";

export const metadata: Metadata = {
  title: "AI-Protect — Guardian Intelligence",
  description: "An Adaptive Adversarial Cloaking & Digital Artwork Research Laboratory. Bounded optimization, cross-architecture transferability, and empirical robustness.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#101217] text-[#F4F1E9] antialiased selection:bg-[#5CE1D2]/25 selection:text-[#5CE1D2]">
        <LabLayout>{children}</LabLayout>
      </body>
    </html>
  );
}
