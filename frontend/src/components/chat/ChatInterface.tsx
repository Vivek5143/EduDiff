"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Message, generateMockResponse } from "@/lib/mock-data";
import { ChatInput } from "./ChatInput";
import { MessageItem } from "./MessageItem";
import { Button } from "@/components/ui/button";

import { Bot, Home } from "lucide-react";

export function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "welcome",
            role: "assistant",
            content: "Hello! I'm EduDiff Lite. Ask me any question, and I'll explain it with text, voice, and visual animations!",
            timestamp: Date.now()
        }
    ]);
    const [isTyping, setIsTyping] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages, isTyping]);

    const handleSend = async (text: string) => {
        const userMsg: Message = {
            id: Date.now().toString(),
            role: "user",
            content: text,
            timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);
        setIsTyping(true);

        try {
            const response = await generateMockResponse(text);
            setMessages(prev => [...prev, response]);
        } catch (error) {
            console.error("Error generating response", error);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-4rem)] md:h-[calc(100vh-2rem)] w-full max-w-5xl mx-auto bg-slate-50/50 rounded-3xl overflow-hidden border border-slate-200/60 shadow-2xl backdrop-blur-sm">

            {/* Header */}
            <div className="p-4 border-b border-slate-100 bg-white/80 flex items-center justify-between shadow-sm z-10">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-md">
                        <Bot className="w-6 h-6" />
                    </div>
                    <div>
                        <h1 className="font-bold text-slate-800">EduDiff Lite</h1>
                        <p className="text-xs text-slate-500">AI-Powered Question-to-Video</p>
                    </div>
                </div>
                <Link href="/">
                    <Button variant="ghost" size="sm" className="text-slate-500 hover:text-slate-900">
                        <Home className="w-4 h-4 mr-2" />
                        Home
                    </Button>
                </Link>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
                {messages.map((msg) => (
                    <MessageItem key={msg.id} message={msg} />
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                    <div className="flex gap-3 items-center text-muted-foreground text-sm ml-2 animate-pulse">
                        <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center">
                            <Bot className="w-4 h-4 text-slate-400" />
                        </div>
                        <span>EduDiff is thinking...</span>
                    </div>
                )}
                <div ref={scrollRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white/50 border-t border-slate-100 backdrop-blur-md">
                <ChatInput onSend={handleSend} disabled={isTyping} />
            </div>
        </div>
    );
}
