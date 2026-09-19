import React, { useEffect, useRef } from "react";
import type { ChatMessage as ChatMessageType } from "../hooks/useChat";
import type { Source } from "../types/api";
import { ChatInput } from "./ChatInput";
import { ChatMessage } from "./ChatMessage";
import { EmptyState } from "./EmptyState";

interface ChatWindowProps {
  messages: ChatMessageType[];
  isSending: boolean;
  onSend: (query: string) => void;
  hasDocuments: boolean;
  isBackendReachable: boolean | null;
  onOpenMobileSidebar: () => void;
  onSelectSource?: (source: Source) => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isSending,
  onSend,
  hasDocuments,
  isBackendReachable,
  onOpenMobileSidebar,
  onSelectSource,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 flex flex-col h-full bg-stone-50 overflow-hidden relative">
      {/* Top Header Bar */}
      <header className="h-14 border-b border-stone-200 bg-white/80 backdrop-blur-xs px-4 flex items-center justify-between shrink-0 z-10">
        <div className="flex items-center gap-3">
          {/* Mobile menu trigger button */}
          <button
            type="button"
            onClick={onOpenMobileSidebar}
            className="md:hidden p-1.5 rounded-md text-stone-600 hover:bg-stone-100 transition-colors"
            aria-label="Open documents drawer"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>

          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold tracking-wide text-stone-900">
              LUNOR
            </span>
            <span className="hidden sm:inline text-xs text-stone-400 font-normal">
              AI Knowledge Assistant
            </span>
          </div>
        </div>

        {/* Connection status indicator */}
        <div className="flex items-center gap-1.5 text-xs font-medium">
          <span
            className={`w-2 h-2 rounded-full ${
              isBackendReachable === true
                ? "bg-emerald-500"
                : isBackendReachable === false
                ? "bg-rose-500"
                : "bg-amber-400 animate-pulse"
            }`}
          />
          <span
            className={
              isBackendReachable === true
                ? "text-stone-600"
                : isBackendReachable === false
                ? "text-rose-600"
                : "text-stone-400"
            }
          >
            {isBackendReachable === true
              ? "Connected"
              : isBackendReachable === false
              ? "Disconnected"
              : "Connecting..."}
          </span>
        </div>
      </header>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 flex flex-col">
        {messages.length === 0 ? (
          <EmptyState
            hasDocuments={hasDocuments}
            onSelectPrompt={onSend}
          />
        ) : (
          <div className="max-w-3xl w-full mx-auto flex-1">
            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                onSelectSource={onSelectSource}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 md:px-8 bg-white border-t border-stone-200/80 shrink-0">
        <div className="max-w-3xl mx-auto w-full">
          <ChatInput
            onSend={onSend}
            isSending={isSending}
            disabled={isBackendReachable === false}
          />
        </div>
      </div>
    </div>
  );
};
