"use client";

import { motion } from 'framer-motion';

export function MediaGrid({ urls }: { urls?: string[] }) {
    if (!urls || urls.length === 0) return null;

    return (
        <div className="grid grid-cols-3 gap-2 mt-3 w-full">
            {urls.map((url, index) => (
                <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 + 0.5 }}
                    className="relative aspect-video rounded-lg overflow-hidden border border-border bg-black/5 shadow-sm group"
                >
                    <img
                        src={url}
                        alt={`Explanation visual ${index + 1}`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                    <div className="absolute bottom-1 right-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded backdrop-blur-sm">
                        GIF {index + 1}
                    </div>
                </motion.div>
            ))}
        </div>
    );
}
