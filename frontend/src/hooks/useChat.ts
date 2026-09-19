import { useCallback, useState } from "react";
import { sendChatMessage } from "../services/api";
import type { Source } from "../types/api";

export type ChatMessageStatus = "sending" | "sent" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  hasSufficientContext?: boolean;
  status: ChatMessageStatus;
}

function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState<boolean>(false);

  const send = useCallback(
    async (query: string) => {
      const trimmedQuery = query.trim();
      if (!trimmedQuery || isSending) {
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

      setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
      setIsSending(true);

      try {
        const response = await sendChatMessage(trimmedQuery);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  content: response.answer,
                  sources: response.sources,
                  hasSufficientContext: response.has_sufficient_context,
                  status: "sent",
                }
              : msg
          )
        );
      } catch (err: unknown) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to get response from assistant";
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  content: errorMessage,
                  status: "error",
                  hasSufficientContext: false,
                }
              : msg
          )
        );
      } finally {
        setIsSending(false);
      }
    },
    [isSending]
  );

  return {
    messages,
    isSending,
    send,
  };
}
