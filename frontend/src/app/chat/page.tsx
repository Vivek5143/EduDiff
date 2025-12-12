import { ChatInterface } from "@/components/chat/ChatInterface";

export default function ChatPage() {
    return (
        <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center p-4 md:p-8">
            <div className="w-full max-w-5xl">
                <ChatInterface />
            </div>
        </main>
    );
}
