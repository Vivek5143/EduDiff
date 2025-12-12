import { Navbar } from "@/components/landing/Navbar";
import { Hero } from "@/components/landing/Hero";
import { FeatureGrid } from "@/components/landing/FeatureGrid";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { DemoSection } from "@/components/landing/DemoSection";

export default function Home() {
  return (
    <main className="min-h-screen bg-white">
      <Navbar />
      <Hero />
      <DemoSection />
      <HowItWorks />
      <FeatureGrid />

      {/* Footer */}
      <footer className="bg-slate-50 py-12 border-t border-slate-100">
        <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
          <p>© 2025 EduDiff Lite. Built for Hackathons.</p>
        </div>
      </footer>
    </main>
  );
}
