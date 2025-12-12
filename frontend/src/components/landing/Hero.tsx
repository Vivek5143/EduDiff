"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { ArrowRight, PlayCircle, Sparkles } from "lucide-react";

export function Hero() {
    return (
        <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
            {/* Background Gradients */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[500px] bg-gradient-to-b from-indigo-50/50 to-transparent -z-10" />
            <div className="absolute top-20 right-0 w-96 h-96 bg-purple-100/50 rounded-full blur-3xl -z-10 opacity-60" />
            <div className="absolute bottom-0 left-0 w-72 h-72 bg-blue-100/50 rounded-full blur-3xl -z-10 opacity-60" />

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-sm font-medium mb-6">
                        <Sparkles className="w-4 h-4" />
                        <span>Now with AI Micro-Animations</span>
                    </div>

                    <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-slate-900 mb-6">
                        Learn Anything with <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-violet-600">
                            Interactive AI
                        </span>
                    </h1>

                    <p className="text-lg md:text-xl text-slate-600 max-w-2xl mx-auto mb-10 leading-relaxed">
                        EduDiff Lite transforms your questions into clear explanations,
                        voice narration, and visual micro-animations.
                        Learning has never been this fast.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link href="/chat">
                            <Button size="lg" className="rounded-full px-8 h-14 text-lg bg-slate-900 hover:bg-slate-800 text-white shadow-xl hover:shadow-2xl transition-all">
                                Start Learning Now <ArrowRight className="w-5 h-5 ml-2" />
                            </Button>
                        </Link>
                        <Link href="#demo">
                            <Button variant="outline" size="lg" className="rounded-full px-8 h-14 text-lg border-slate-200 hover:bg-slate-50 text-slate-700">
                                <PlayCircle className="w-5 h-5 mr-2" /> Watch Demo
                            </Button>
                        </Link>
                    </div>
                </motion.div>
            </div>
        </section>
    );
}
