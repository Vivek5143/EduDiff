"use client";

import { motion } from "framer-motion";

import { useEffect, useState } from "react";
import { api, DemoVideo } from "@/lib/api";

export function DemoSection() {
    const [videos, setVideos] = useState<DemoVideo[]>([]);

    useEffect(() => {
        api.getDemos().then(setVideos);
    }, []);

    return (
        <section id="demo" className="py-24 bg-white relative">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">See EduDiff in Action</h2>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Watch how EduDiff Lite transforms mathematical concepts into visual animations.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl mx-auto">
                    {videos.length > 0 ? (
                        videos.map((video, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.1 }}
                                className="bg-white rounded-xl overflow-hidden shadow-lg border border-slate-200"
                            >
                                <div className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-slate-200">
                                    <h3 className="text-xl font-semibold text-slate-900">{video.title}</h3>
                                </div>
                                <div className="aspect-video bg-slate-900 flex items-center justify-center">
                                    <img
                                        src={video.url}
                                        alt={video.title}
                                        className="w-full h-full object-contain"
                                    />
                                </div>
                                <div className="p-4 bg-slate-50">
                                    <p className="text-sm text-slate-600">{video.description}</p>
                                </div>
                            </motion.div>
                        ))
                    ) : (
                        <div className="col-span-full relative aspect-video rounded-3xl overflow-hidden shadow-2xl bg-slate-900 max-w-4xl mx-auto">
                            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-indigo-900 to-purple-900">
                                <span className="text-white text-lg opacity-70">Loading demo animations...</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}
