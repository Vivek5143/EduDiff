export type MessageRole = 'user' | 'assistant';

export interface Message {
    id: string;
    role: MessageRole;
    content: string;
    audioUrl?: string; // URL to Mock Audio
    videoUrl?: string; // URL to generated video
    visualUrls?: string[]; // URLs to Mock GIFs
    timestamp: number;
}

export const MOCK_EXPLANATIONS = {
    default: {
        text: "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (c) is equal to the sum of the squares of the other two sides (a and b). Formula: a² + b² = c².",
        audio: "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav", // Placeholder
        gifs: [
            "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif", // Math/Geometry placeholder
            "https://media.giphy.com/media/l0HlJDaeqNXVcOBuk/giphy.gif", // Cool visual
            "https://media.giphy.com/media/dh5pCjG8PUCjk5uWjM/giphy.gif" // Calculation
        ]
    }
};

export async function generateMockResponse(question: string): Promise<Message> {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                id: Date.now().toString(),
                role: 'assistant',
                content: MOCK_EXPLANATIONS.default.text,
                audioUrl: MOCK_EXPLANATIONS.default.audio,
                visualUrls: MOCK_EXPLANATIONS.default.gifs,
                timestamp: Date.now()
            });
        }, 1500); // Simulate 1.5s latency
    });
}
