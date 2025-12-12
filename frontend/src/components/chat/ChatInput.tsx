"use client";

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Sparkles } from "lucide-react";
import { motion } from "framer-motion";

interface ChatInputProps {
    onSend: (message: string) => void;
    disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
    const [input, setInput] = useState("");

    const handleSend = () => {
        if (input.trim()) {
            onSend(input);
            setInput("");
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="relative flex items-center w-full max-w-4xl mx-auto p-4 bg-white/80 backdrop-blur-md rounded-2xl border border-primary/20 shadow-xl">
            <Sparkles className="w-5 h-5 text-primary mr-3 animate-pulse" />
            <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question to learn with visuals..."
                disabled={disabled}
                className="flex-1 border-none shadow-none focus-visible:ring-0 text-base py-6 bg-transparent placeholder:text-muted-foreground/50"
            />
            <Button
                onClick={handleSend}
                disabled={disabled || !input.trim()}
                size="icon"
                className="ml-2 rounded-xl bg-primary hover:bg-primary/90 transition-all duration-300 shadow-lg shadow-primary/20"
            >
                <Send className="w-5 h-5 text-primary-foreground" />
            </Button>
        </div>
    );
}
