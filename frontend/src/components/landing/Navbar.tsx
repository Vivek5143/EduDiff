"use client";

import Link from "next/link";
import { Bot } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Navbar() {
    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-indigo-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <Link href="/" className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-600 to-violet-600 flex items-center justify-center text-white shadow-lg shadow-indigo-200">
                            <Bot className="w-5 h-5" />
                        </div>
                        <span className="font-bold text-xl tracking-tight text-slate-900">EduDiff Lite</span>
                    </Link>

                    <div className="hidden md:flex items-center gap-6">
                        <Link href="#features" className="text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors">Features</Link>
                        <Link href="#how-it-works" className="text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors">How it Works</Link>
                        <Link href="/chat">
                            <Button className="rounded-full px-6 bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 shadow-lg">
                                Try Demo
                            </Button>
                        </Link>
                    </div>
                </div>
            </div>
        </nav>
    );
}
