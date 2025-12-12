"use client";

import { motion } from "framer-motion";
import { Search, Brain, Rocket } from "lucide-react";

export function HowItWorks() {
    const steps = [
        {
            icon: Search,
            title: "1. Ask a Question",
            description: "Type any question you have, from 'How does photosynthesis work?' to complex calculus problems."
        },
        {
            icon: Brain,
            title: "2. AI Generates Explanation",
            description: "Our AI instantly analyzes your query and creates a custom explanation with text, voice, and visuals."
        },
        {
            icon: Rocket,
            title: "3. Learn Faster",
            description: "Absorb the information quickly through our multi-modal format and retain it better."
        }
    ];

    return (
        <section id="how-it-works" className="py-24 bg-slate-50 relative overflow-hidden">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">How It Works</h2>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Three simple steps to supercharge your learning.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.2 }}
                            className="relative flex flex-col items-center text-center p-8 bg-white rounded-2xl shadow-sm border border-slate-100"
                        >
                            <div className="w-16 h-16 rounded-full bg-indigo-100 flex items-center justify-center mb-6 text-indigo-600">
                                <step.icon className="w-8 h-8" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-4">{step.title}</h3>
                            <p className="text-slate-600">
                                {step.description}
                            </p>

                            {/* Connector Line (Desktop only) */}
                            {index < steps.length - 1 && (
                                <div className="hidden md:block absolute top-1/2 -right-4 w-8 h-0.5 bg-slate-200 transform translate-x-1/2" />
                            )}
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
