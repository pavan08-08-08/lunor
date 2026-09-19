import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { ChatMessage } from "../components/ChatMessage";
import { DocumentList } from "../components/DocumentList";
import { EmptyState } from "../components/EmptyState";
import { SourceList } from "../components/SourceList";
import { UploadButton } from "../components/UploadButton";
import { useChat } from "../hooks/useChat";
import {
  ApiError,
  getDocuments,
  getHealth,
} from "../services/api";

describe("Frontend Unit Tests", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // 1. API service success
  it("API service handles successful responses correctly", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" }),
    });

    const health = await getHealth();
    expect(health).toEqual({ status: "ok" });

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        documents: [{ filename: "doc1.pdf" }, { filename: "doc2.pdf" }],
      }),
    });

    const docs = await getDocuments();
    expect(docs.documents).toHaveLength(2);
    expect(docs.documents[0].filename).toBe("doc1.pdf");
  });

  // 2. API service HTTP error
  it("API service throws ApiError with detail on HTTP failure", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Invalid file format" }),
    });

    await expect(getDocuments()).rejects.toThrow("Invalid file format");
    await expect(getDocuments()).rejects.toBeInstanceOf(ApiError);
  });

  // 3. EmptyState no-documents
  it("EmptyState renders upload prompt when no documents are indexed", () => {
    render(<EmptyState hasDocuments={false} />);
    expect(
      screen.getByText("Upload a PDF in the sidebar to start asking grounded questions.")
    ).toBeInTheDocument();
  });

  // 4. EmptyState welcome
  it("EmptyState renders welcome message and suggested questions when documents exist", () => {
    const handleSelect = vi.fn();
    render(<EmptyState hasDocuments={true} onSelectPrompt={handleSelect} />);
    expect(screen.getByText("Lunor Knowledge Assistant")).toBeInTheDocument();
    expect(
      screen.getByText("Ask questions grounded directly in your uploaded PDF documents.")
    ).toBeInTheDocument();

    const sampleBtn = screen.getByText("What are the main findings of the paper?");
    expect(sampleBtn).toBeInTheDocument();
    fireEvent.click(sampleBtn);
    expect(handleSelect).toHaveBeenCalledWith(
      "What are the main findings of the paper?"
    );
  });

  // 5. DocumentList rendering
  it("DocumentList renders empty state and list of documents correctly", () => {
    const { rerender } = render(
      <DocumentList documents={[]} isLoading={false} />
    );
    expect(screen.getByText("No documents yet")).toBeInTheDocument();

    rerender(
      <DocumentList
        documents={[{ filename: "paper.pdf" }, { filename: "research.pdf" }]}
        isLoading={false}
      />
    );
    expect(screen.getByText("paper.pdf")).toBeInTheDocument();
    expect(screen.getByText("research.pdf")).toBeInTheDocument();
    expect(screen.getByText("Indexed Files (2)")).toBeInTheDocument();
  });

  // 6. Chat state success
  it("useChat handles successful message dispatch and response update", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: "This is the generated answer.",
        sources: [{ source_filename: "doc.pdf", page_number: 2 }],
        has_sufficient_context: true,
      }),
    });

    function TestChatComponent() {
      const { messages, isSending, send } = useChat();
      return (
        <div>
          <button onClick={() => send("What is the thesis?")}>Ask</button>
          <div data-testid="is-sending">{String(isSending)}</div>
          <div data-testid="messages-count">{messages.length}</div>
          {messages.map((m) => (
            <div key={m.id} data-testid={`msg-${m.role}`}>
              {m.content}
            </div>
          ))}
        </div>
      );
    }

    render(<TestChatComponent />);
    const askButton = screen.getByText("Ask");

    await act(async () => {
      fireEvent.click(askButton);
    });

    await waitFor(() => {
      expect(screen.getByTestId("msg-user")).toHaveTextContent("What is the thesis?");
      expect(screen.getByTestId("msg-assistant")).toHaveTextContent(
        "This is the generated answer."
      );
    });
  });

  // 7. Chat state error
  it("useChat sets assistant message to error status on network failure", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("Network connection dropped"));

    function TestChatErrorComponent() {
      const { messages, send } = useChat();
      return (
        <div>
          <button onClick={() => send("Faulty query")}>Ask</button>
          {messages.map((m) => (
            <div key={m.id} data-testid={`msg-${m.role}-${m.status}`}>
              {m.content}
            </div>
          ))}
        </div>
      );
    }

    render(<TestChatErrorComponent />);
    const askButton = screen.getByText("Ask");

    await act(async () => {
      fireEvent.click(askButton);
    });

    await waitFor(() => {
      expect(screen.getByTestId("msg-assistant-error")).toHaveTextContent(
        "Network connection dropped"
      );
    });
  });

  // 8. SourceList rendering
  it("SourceList renders source filename and page numbers", () => {
    const sources = [
      { source_filename: "traffic_study.pdf", page_number: 1 },
      { source_filename: "traffic_study.pdf", page_number: 4 },
    ];

    render(<SourceList sources={sources} />);
    expect(screen.getAllByText("traffic_study.pdf")).toHaveLength(2);
    expect(screen.getByText("Page 1")).toBeInTheDocument();
    expect(screen.getByText("Page 4")).toBeInTheDocument();
  });

  // 9. Upload failure state
  it("UploadButton displays error message on failure", async () => {
    const mockUpload = vi.fn().mockResolvedValue(false);

    render(
      <UploadButton
        onUpload={mockUpload}
        uploadState="error"
        uploadError="Only PDF files are allowed"
      />
    );

    expect(screen.getByText("Only PDF files are allowed")).toBeInTheDocument();
  });

  // 10. ChatMessage sources displayed when hasSufficientContext=true
  it("ChatMessage renders sources when hasSufficientContext is true", () => {
    const message = {
      id: "msg-1",
      role: "assistant" as const,
      content: "The framework proposes XGBoost and Random Forest.",
      sources: [{ source_filename: "traffic_study.pdf", page_number: 2 }],
      hasSufficientContext: true,
      status: "sent" as const,
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText("The framework proposes XGBoost and Random Forest.")).toBeInTheDocument();
    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("traffic_study.pdf")).toBeInTheDocument();
    expect(screen.getByText("Page 2")).toBeInTheDocument();
    expect(screen.queryByText("Limited context")).not.toBeInTheDocument();
  });

  // 11. ChatMessage sources NOT displayed when hasSufficientContext=false
  it("ChatMessage does NOT render sources when hasSufficientContext is false", () => {
    const message = {
      id: "msg-2",
      role: "assistant" as const,
      content: "Based on the provided context, there is no information about who the CEO of Project Atlas is.",
      sources: [{ source_filename: "candidate_chunk.pdf", page_number: 5 }],
      hasSufficientContext: false,
      status: "sent" as const,
    };

    render(<ChatMessage message={message} />);
    expect(
      screen.getByText(
        "Based on the provided context, there is no information about who the CEO of Project Atlas is."
      )
    ).toBeInTheDocument();
    expect(screen.queryByText("Sources")).not.toBeInTheDocument();
    expect(screen.queryByText("candidate_chunk.pdf")).not.toBeInTheDocument();
    expect(screen.queryByText("Page 5")).not.toBeInTheDocument();
    expect(screen.getByText("Limited context")).toBeInTheDocument();
  });
});
