"use client";

import { motion } from "framer-motion";
import { Mic, Image as ImageIcon, FileText, Zap } from "lucide-react";

const features = [
    {
        icon: FileText,
        title: "Clear Explanations",
        description: "Get structured, easy-to-understand text answers generated instantly by AI.",
        color: "bg-blue-50 text-blue-600"
    },
    {
        icon: Mic,
        title: "Voice Narration",
        description: "Listen to natural-sounding audio explanations, just like a real tutor.",
        color: "bg-purple-50 text-purple-600"
    },
    {
        icon: ImageIcon,
        title: "Visual Micro-GIFs",
        description: "Understand complex concepts with automatically generated 3-second animations.",
        color: "bg-pink-50 text-pink-600"
    },
    {
        icon: Zap,
        title: "Instant Results",
        description: "No long video rendering. Get your micro-lessons in seconds.",
        color: "bg-amber-50 text-amber-600"
    }
];

export function FeatureGrid() {
    return (
        <section id="features" className="py-24 bg-white relative">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Everything You Need to Learn</h2>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        We combine three powerful learning modalities into one simple interface.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {features.map((feature, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            className="p-6 rounded-2xl border border-slate-100 bg-white hover:shadow-xl hover:-translate-y-1 transition-all duration-300"
                        >
                            <div className={`w-12 h-12 rounded-xl ${feature.color} flex items-center justify-center mb-6`}>
                                <feature.icon className="w-6 h-6" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">{feature.title}</h3>
                            <p className="text-slate-600 leading-relaxed">
                                {feature.description}
                            </p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
