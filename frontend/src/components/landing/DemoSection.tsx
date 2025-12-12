"use client";

import { motion } from "framer-motion";
import { Play } from "lucide-react";

export function DemoSection() {
    return (
        <section id="demo" className="py-24 bg-white relative">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">See EduDiff in Action</h2>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Watch how EduDiff Lite transforms a simple question into a comprehensive lesson.
                    </p>
                </div>

                <div className="relative max-w-4xl mx-auto aspect-video rounded-3xl overflow-hidden shadow-2xl bg-slate-900 group cursor-pointer">
                    {/* Placeholder for actual video */}
                    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-indigo-900/50 to-purple-900/50">
                        <motion.div
                            whileHover={{ scale: 1.1 }}
                            className="w-20 h-20 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center text-white border-2 border-white/50"
                        >
                            <Play className="w-8 h-8 ml-1 fill-white" />
                        </motion.div>
                    </div>

                    <div className="absolute inset-0 flex items-end justify-center pb-8 opacity-0 group-hover:opacity-100 transition-opacity">
                        <span className="text-white font-medium bg-black/50 px-4 py-2 rounded-full backdrop-blur-md">
                            Click to Watch Demo (Placeholder)
                        </span>
                    </div>

                    {/* Simulated Interface Background */}
                    <div className="w-full h-full opacity-30 flex items-center justify-center">
                        <div className="text-slate-50 text-9xl font-bold opacity-10 select-none">
                            DEMO
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
