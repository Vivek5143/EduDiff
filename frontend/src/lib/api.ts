const API_URL = 'http://localhost:5001';

export interface GenerateResponse {
    success: boolean;
    video_url: string;
    explanation: string;
    code: string;
    used_ai: boolean;
}

export interface DemoVideo {
    filename: string;
    title: string;
    description: string;
    url: string;
}

export const api = {
    async generateVideo(concept: string): Promise<GenerateResponse> {
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ concept }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || 'Failed to generate video');
        }

        const data = await response.json();
        // Convert relative URL to absolute
        if (data.video_url && !data.video_url.startsWith('http')) {
            data.video_url = `${API_URL}${data.video_url}`;
        }
        return data;
    },

    async getDemos(): Promise<DemoVideo[]> {
        try {
            const response = await fetch(`${API_URL}/demos`);
            if (!response.ok) {
                return [];
            }
            const data = await response.json();
            // Convert relative URLs to absolute
            const videos = (data.videos || []).map((video: DemoVideo) => ({
                ...video,
                url: video.url.startsWith('http') ? video.url : `${API_URL}${video.url}`
            }));
            return videos;
        } catch (error) {
            console.error("Failed to fetch demos:", error);
            return [];
        }
    }
};
