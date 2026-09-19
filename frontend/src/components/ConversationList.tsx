import React from "react";
import type { Conversation } from "../types/api";

interface ConversationListProps {
  conversations: Conversation[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  disabled?: boolean;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  activeConversationId,
  onSelectConversation,
  disabled = false,
}) => {
  if (conversations.length === 0) {
    return (
      <div className="px-3 py-2 text-xs text-stone-400 italic">
        No conversations yet
      </div>
    );
  }

  return (
    <div className="space-y-0.5" role="list" aria-label="Conversation history">
      {conversations.map((conv) => {
        const isActive = conv.id === activeConversationId;
        return (
          <button
            key={conv.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelectConversation(conv.id)}
            className={`w-full text-left px-2.5 py-1.5 rounded-md text-xs flex items-center gap-2 transition-colors cursor-pointer group ${
              isActive
                ? "bg-brand-50 text-brand-700 font-medium border border-brand-200/70 shadow-xs"
                : "text-stone-600 hover:bg-stone-100 hover:text-stone-900 border border-transparent"
            }`}
            title={conv.title}
          >
            <span className="text-xs shrink-0 text-stone-400 group-hover:text-stone-600" aria-hidden="true">
              💬
            </span>
            <span className="truncate flex-1">{conv.title}</span>
          </button>
        );
      })}
    </div>
  );
};
