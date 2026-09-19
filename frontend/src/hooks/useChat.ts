import { useCallback, useState } from "react";
import { ApiError, sendChatMessage } from "../services/api";
import type {
  ChatMessage,
  ChatMessageStatus,
  Conversation,
} from "../types/api";

export type { ChatMessage, ChatMessageStatus, Conversation };

export function formatConversationTitle(text: string, maxLength: number = 50): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return "New conversation";
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return normalized.slice(0, maxLength).trim() + "...";
}

function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `id_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function createNewConversation(title: string = "New conversation"): Conversation {
  const now = Date.now();
  return {
    id: generateId(),
    title,
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

export function useChat() {
  const [initial] = useState(() => createNewConversation());
  const [conversations, setConversations] = useState<Conversation[]>([initial]);
  const [activeConversationId, setActiveConversationId] = useState<string>(initial.id);
  const [isSending, setIsSending] = useState<boolean>(false);

  const activeConversation =
    conversations.find((c) => c.id === activeConversationId) || conversations[0];

  const messages = activeConversation ? activeConversation.messages : [];

  const createConversation = useCallback(() => {
    // If the active conversation is already empty, keep it active instead of creating unlimited empties
    const current = conversations.find((c) => c.id === activeConversationId);
    if (current && current.messages.length === 0) {
      return;
    }

    // If another existing conversation is empty, switch to it
    const existingEmpty = conversations.find((c) => c.messages.length === 0);
    if (existingEmpty) {
      setActiveConversationId(existingEmpty.id);
      return;
    }

    const newConv = createNewConversation();
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
  }, [conversations, activeConversationId]);

  const selectConversation = useCallback((id: string) => {
    setActiveConversationId(id);
  }, []);

  const send = useCallback(
    async (query: string) => {
      const trimmedQuery = query.trim();
      if (!trimmedQuery || isSending) {
        return;
      }

      const targetConversationId = activeConversation?.id || activeConversationId;
      if (!targetConversationId) {
        return;
      }

      const userMsgId = generateId();
      const assistantMsgId = generateId();

      const userMessage: ChatMessage = {
        id: userMsgId,
        role: "user",
        content: trimmedQuery,
        status: "sent",
      };

      const assistantPlaceholder: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        status: "sending",
      };

      const now = Date.now();

      // Append user message + assistant placeholder, and update title on first message
      setConversations((prev) =>
        prev.map((conv) => {
          if (conv.id !== targetConversationId) return conv;

          const isFirstMessage =
            conv.messages.length === 0 || conv.title === "New conversation";
          const updatedTitle = isFirstMessage
            ? formatConversationTitle(trimmedQuery)
            : conv.title;

          return {
            ...conv,
            title: updatedTitle,
            messages: [...conv.messages, userMessage, assistantPlaceholder],
            updatedAt: now,
          };
        })
      );

      setIsSending(true);

      try {
        const response = await sendChatMessage(trimmedQuery);
        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id !== targetConversationId) return conv;

            return {
              ...conv,
              updatedAt: Date.now(),
              messages: conv.messages.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      content: response.answer,
                      sources: response.sources,
                      hasSufficientContext: response.has_sufficient_context,
                      status: "sent",
                    }
                  : msg
              ),
            };
          })
        );
      } catch (err: unknown) {
        let errorMessage =
          err instanceof Error ? err.message : "Failed to get response from assistant";

        if (
          (err instanceof ApiError && err.status === 429) ||
          ((err as { status?: number })?.status === 429)
        ) {
          errorMessage = "Gemini API quota exhausted. Please try again after the quota resets.";
        }

        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id !== targetConversationId) return conv;

            return {
              ...conv,
              updatedAt: Date.now(),
              messages: conv.messages.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      content: errorMessage,
                      status: "error",
                      hasSufficientContext: false,
                    }
                  : msg
              ),
            };
          })
        );
      } finally {
        setIsSending(false);
      }
    },
    [activeConversation, activeConversationId, isSending]
  );

  return {
    conversations,
    activeConversationId: activeConversation?.id || activeConversationId,
    activeConversation,
    messages,
    isSending,
    send,
    createConversation,
    selectConversation,
  };
}
