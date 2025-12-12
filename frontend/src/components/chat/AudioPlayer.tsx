"use client";

import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2 } from 'lucide-react';
import { Button } from '@/components/ui/button';


// Note: Shadcn Slider is not installed yet, I used basic div bar for now or I should install slider?
// I'll implement a simple fake progress bar for now to avoid dependency error if slider is missing.
// Actually, I didn't install slider. I'll use a simple progress div.

export function AudioPlayer({ src }: { src: string }) {
    const [isPlaying, setIsPlaying] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    useEffect(() => {
        if (audioRef.current) {
            if (isPlaying) audioRef.current.play();
            else audioRef.current.pause();
        }
    }, [isPlaying]);

    return (
        <div className="flex items-center gap-3 p-3 bg-secondary/50 rounded-xl border border-secondary w-full max-w-md my-2">
            <audio
                ref={audioRef}
                src={src}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
            />
            <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground shrink-0 shadow-md"
                onClick={() => setIsPlaying(!isPlaying)}
            >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-1" />}
            </Button>

            <div className="flex-1 flex flex-col gap-1">
                <div className="text-xs font-semibold text-primary uppercase tracking-wider">AI Narration</div>
                <div className="h-1.5 w-full bg-primary/20 rounded-full overflow-hidden">
                    <div className={`h-full bg-primary rounded-full transition-all duration-[2000ms] ${isPlaying ? 'w-full' : 'w-0'}`} />
                    {/* Fake animation for now */}
                </div>
            </div>

            <div className="p-2 bg-white rounded-full text-primary shadow-sm">
                <Volume2 className="w-4 h-4" />
            </div>
        </div>
    );
}
