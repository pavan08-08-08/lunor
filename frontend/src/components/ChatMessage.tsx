import React from "react";
import type { ChatMessage as ChatMessageType } from "../hooks/useChat";
import type { Source } from "../types/api";
import { SourceList } from "./SourceList";

interface ChatMessageProps {
  message: ChatMessageType;
  onSelectSource?: (source: Source) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onSelectSource }) => {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[85%] md:max-w-[75%] rounded-2xl rounded-tr-sm bg-brand-600 text-white px-4 py-2.5 shadow-sm">
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  // Assistant Message
  return (
    <div className="flex justify-start mb-4">
      <div
        className={`max-w-[85%] md:max-w-[75%] rounded-2xl rounded-tl-sm bg-white border px-4 py-3 shadow-sm ${
          message.status === "error"
            ? "border-rose-200 bg-rose-50/50 text-rose-900"
            : "border-stone-200 text-stone-800"
        }`}
      >
        {/* Assistant Header / Limited Context Flag */}
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <span className="text-[11px] font-semibold tracking-wider text-stone-400 uppercase">
            Lunor
          </span>
          {message.hasSufficientContext === false && message.status === "sent" && (
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 bg-amber-50 border border-amber-200/60 rounded px-1.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
              Limited context
            </span>
          )}
        </div>

        {/* Loading Indicator */}
        {message.status === "sending" ? (
          <div className="py-2 flex items-center space-x-1.5" aria-label="Thinking">
            <span className="w-2 h-2 rounded-full bg-brand-600 animate-dot-1" />
            <span className="w-2 h-2 rounded-full bg-brand-600 animate-dot-2" />
            <span className="w-2 h-2 rounded-full bg-brand-600 animate-dot-3" />
          </div>
        ) : (
          <div className="text-sm whitespace-pre-wrap leading-relaxed">
            {message.content}
          </div>
        )}

        {/* Source Citations - only rendered when context is sufficient */}
        {message.status === "sent" && message.hasSufficientContext === true && message.sources && (
          <SourceList sources={message.sources} onSelectSource={onSelectSource} />
        )}
      </div>
    </div>
  );
};
