"use client";

import { Message } from "@/lib/mock-data";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { AudioPlayer } from "./AudioPlayer";
import { MediaGrid } from "./MediaGrid";
import { motion } from "framer-motion";
import { Bot, User } from "lucide-react";

interface MessageItemProps {
    message: Message;
}

export function MessageItem({ message }: MessageItemProps) {
    const isUser = message.role === "user";

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
                "flex w-full mb-6",
                isUser ? "justify-end" : "justify-start"
            )}
        >
            <div className={cn("flex max-w-[85%] md:max-w-[70%] gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
                {/* Avatar */}
                <Avatar className="w-8 h-8 md:w-10 md:h-10 border border-border shadow-sm">
                    {isUser ? (
                        <AvatarFallback className="bg-primary text-primary-foreground"><User className="w-5 h-5" /></AvatarFallback>
                    ) : (
                        <AvatarFallback className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white"><Bot className="w-6 h-6" /></AvatarFallback>
                    )}
                </Avatar>

                {/* Bubble */}
                <div className={cn(
                    "flex flex-col p-4 rounded-2xl shadow-sm border",
                    isUser
                        ? "bg-primary text-primary-foreground rounded-tr-sm border-primary/20"
                        : "bg-white text-slate-800 rounded-tl-sm border-slate-100"
                )}>
                    {/* Text Content */}
                    <p className="text-sm md:text-base leading-relaxed whitespace-pre-wrap">
                        {message.content}
                    </p>

                    {/* AI Assets */}
                    {!isUser && (
                        <>
                            {/* Audio Player */}
                            {message.audioUrl && (
                                <div className="mt-3">
                                    <AudioPlayer src={message.audioUrl} />
                                </div>
                            )}

                            {/* Video Player */}
                            {message.videoUrl && (
                                <div className="mt-3 rounded-lg overflow-hidden border border-border shadow-sm">
                                    <video
                                        src={message.videoUrl}
                                        controls
                                        className="w-full aspect-video bg-black"
                                    />
                                </div>
                            )}

                            {/* Visuals */}
                            {message.visualUrls && (
                                <div className="mt-2">
                                    <MediaGrid urls={message.visualUrls} />
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </motion.div>
    );
}
