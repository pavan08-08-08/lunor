import React, { useState } from "react";

interface ChatInputProps {
  onSend: (query: string) => void;
  isSending: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, isSending, disabled = false }) => {
  const [query, setQuery] = useState("");

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isSending || disabled) return;

    onSend(trimmed);
    setQuery("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const canSubmit = query.trim().length > 0 && !isSending && !disabled;

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <div className="relative flex items-center bg-white rounded-xl border border-stone-300 shadow-sm focus-within:border-brand-500 focus-within:ring-1 focus-within:ring-brand-500 transition-all overflow-hidden">
        <textarea
          rows={1}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? "Backend unavailable..." : "Ask about your documents..."}
          disabled={disabled || isSending}
          className="w-full resize-none py-3.5 pl-4 pr-12 text-sm text-stone-900 placeholder-stone-400 bg-transparent focus:outline-none max-h-32"
          aria-label="Ask a question about your documents"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          aria-label="Send question"
          className={`absolute right-2 p-2 rounded-lg transition-colors flex items-center justify-center ${
            canSubmit
              ? "bg-brand-600 hover:bg-brand-700 text-white cursor-pointer"
              : "bg-stone-100 text-stone-300 cursor-not-allowed"
          }`}
        >
          <svg
            className="w-4 h-4 transform rotate-90"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
          </svg>
        </button>
      </div>
      <div className="mt-1.5 px-1 flex items-center justify-between text-[11px] text-stone-400">
        <span>Press <kbd className="font-mono bg-stone-100 px-1 rounded border border-stone-200">Enter ↵</kbd> to send, <kbd className="font-mono bg-stone-100 px-1 rounded border border-stone-200">Shift+Enter</kbd> for newline</span>
      </div>
    </form>
  );
};
