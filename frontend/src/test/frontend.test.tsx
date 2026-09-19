import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { ChatMessage } from "../components/ChatMessage";
import { DocumentList } from "../components/DocumentList";
import { EmptyState } from "../components/EmptyState";
import { PdfViewerPanel } from "../components/PdfViewerPanel";
import { SourceList } from "../components/SourceList";
import { UploadButton } from "../components/UploadButton";
import { useChat } from "../hooks/useChat";
import {
  ApiError,
  deleteDocument,
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
        sources: [
          {
            source_filename: "doc.pdf",
            page_number: 2,
            chunk_id: "doc_p2_c0",
            text: "Sample evidence text",
          },
        ],
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
      {
        source_filename: "traffic_study.pdf",
        page_number: 1,
        chunk_id: "ts_p1_c0",
        text: "Traffic density rises at peak hours.",
      },
      {
        source_filename: "traffic_study.pdf",
        page_number: 4,
        chunk_id: "ts_p4_c1",
        text: "Signal timing reduces intersection delays.",
      },
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
      sources: [
        {
          source_filename: "traffic_study.pdf",
          page_number: 2,
          chunk_id: "ts_p2_c0",
          text: "XGBoost models show superior prediction accuracy.",
        },
      ],
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
      sources: [
        {
          source_filename: "candidate_chunk.pdf",
          page_number: 5,
          chunk_id: "cand_p5_c0",
          text: "Generic candidate chunk text.",
        },
      ],
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

  // 12. Chat error handling for 429 quota exhaustion
  it("useChat sets assistant message to quota error on HTTP 429", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({
        detail: "Gemini API quota exhausted. Please try again after the quota resets.",
      }),
    });

    function TestChatQuotaComponent() {
      const { messages, send } = useChat();
      return (
        <div>
          <button onClick={() => send("What is the speed?")}>Ask</button>
          {messages.map((m) => (
            <div key={m.id} data-testid={`msg-${m.role}-${m.status}`}>
              {m.content}
            </div>
          ))}
        </div>
      );
    }

    render(<TestChatQuotaComponent />);
    const askButton = screen.getByText("Ask");

    await act(async () => {
      fireEvent.click(askButton);
    });

    await waitFor(() => {
      expect(screen.getByTestId("msg-assistant-error")).toHaveTextContent(
        "Gemini API quota exhausted. Please try again after the quota resets."
      );
    });
  });

  // 13. deleteDocument API function handles successful response and encodes URI
  it("deleteDocument API function sends DELETE request and returns response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        message: "Document deleted successfully",
        filename: "my report.pdf",
        remaining_documents: 1,
      }),
    });

    const result = await deleteDocument("my report.pdf");
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/documents/my%20report.pdf"),
      expect.objectContaining({ method: "DELETE" })
    );
    expect(result.remaining_documents).toBe(1);
    expect(result.filename).toBe("my report.pdf");
  });

  // 14. DocumentList renders delete buttons with aria-label
  it("DocumentList renders delete button with accessible aria-label", () => {
    const handleDelete = vi.fn();
    render(
      <DocumentList
        documents={[{ filename: "atlas.pdf" }, { filename: "borealis.pdf" }]}
        isLoading={false}
        onDelete={handleDelete}
      />
    );

    const deleteAtlasBtn = screen.getByRole("button", { name: "Delete atlas.pdf" });
    const deleteBorealisBtn = screen.getByRole("button", { name: "Delete borealis.pdf" });
    expect(deleteAtlasBtn).toBeInTheDocument();
    expect(deleteBorealisBtn).toBeInTheDocument();

    fireEvent.click(deleteAtlasBtn);
    expect(handleDelete).toHaveBeenCalledTimes(1);
    expect(handleDelete).toHaveBeenCalledWith("atlas.pdf");
  });

  // 15. DocumentList disables button during deletion
  it("DocumentList disables delete button when deletingFilename matches", () => {
    const handleDelete = vi.fn();
    render(
      <DocumentList
        documents={[{ filename: "atlas.pdf" }, { filename: "borealis.pdf" }]}
        isLoading={false}
        onDelete={handleDelete}
        deletingFilename="atlas.pdf"
      />
    );

    const deleteAtlasBtn = screen.getByRole("button", { name: "Delete atlas.pdf" });
    const deleteBorealisBtn = screen.getByRole("button", { name: "Delete borealis.pdf" });

    expect(deleteAtlasBtn).toBeDisabled();
    expect(deleteBorealisBtn).not.toBeDisabled();
  });

  // 16. Source citation renders as accessible button and triggers onSelectSource
  it("Source citation renders as accessible button and triggers onSelectSource", () => {
    const handleSelectSource = vi.fn();
    const source = {
      source_filename: "01_Project_Atlas.pdf",
      page_number: 1,
      chunk_id: "atlas_p1_c0",
      text: "Project Atlas achieved recall@5 = 0.82.",
    };

    render(<SourceList sources={[source]} onSelectSource={handleSelectSource} />);

    const citationBtn = screen.getByRole("button", {
      name: "View evidence in 01_Project_Atlas.pdf, page 1",
    });
    expect(citationBtn).toBeInTheDocument();

    fireEvent.click(citationBtn);
    expect(handleSelectSource).toHaveBeenCalledTimes(1);
    expect(handleSelectSource).toHaveBeenCalledWith(source);
  });

  // 17. PdfViewerPanel renders filename, page indicator, and exact evidence text
  it("PdfViewerPanel renders filename, page indicator, and exact evidence text", async () => {
    const handleClose = vi.fn();
    const source = {
      source_filename: "01_Project_Atlas.pdf",
      page_number: 1,
      chunk_id: "atlas_p1_c0",
      text: "internal experiment recall@5 = 0.82",
      evidence_text: "Atlas achieved a retrieval recall@5 of 0.82.",
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        filename: "01_Project_Atlas.pdf",
        page_number: 1,
        total_pages: 5,
        page_width: 595,
        page_height: 841,
        svg: "<svg><text>Atlas Page Content</text></svg>",
        evidence_text: "Atlas achieved a retrieval recall@5 of 0.82.",
        highlights: [{ x: 100, y: 200, width: 150, height: 20 }],
      }),
    });

    render(
      <PdfViewerPanel
        source={source}
        onClose={handleClose}
        isDocumentAvailable={true}
      />
    );

    expect(screen.getByText("01_Project_Atlas.pdf")).toBeInTheDocument();
    expect(screen.getByText("Page 1")).toBeInTheDocument();
    expect(screen.getByText("Atlas achieved a retrieval recall@5 of 0.82.")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Page 1 of 5")).toBeInTheDocument();
      expect(screen.getByTestId("evidence-highlight")).toBeInTheDocument();
    });
  });

  // 18. PdfViewerPanel navigation and zoom controls function properly
  it("PdfViewerPanel navigation and zoom controls function properly", async () => {
    const handleClose = vi.fn();
    const source = {
      source_filename: "report.pdf",
      page_number: 1,
      chunk_id: "rep_p1",
      text: "Evidence text.",
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        filename: "report.pdf",
        page_number: 1,
        total_pages: 3,
        page_width: 600,
        page_height: 800,
        svg: "<svg><text>Report Page 1</text></svg>",
        evidence_text: "Evidence text.",
        highlights: [],
      }),
    });

    render(
      <PdfViewerPanel
        source={source}
        onClose={handleClose}
        isDocumentAvailable={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
    });

    // Zoom controls
    const zoomInBtn = screen.getByRole("button", { name: "Zoom in" });
    const zoomOutBtn = screen.getByRole("button", { name: "Zoom out" });
    const fitPageBtn = screen.getByRole("button", { name: "Fit to page" });
    const fitWidthBtn = screen.getByRole("button", { name: "Fit to width" });

    expect(screen.getByText("100%")).toBeInTheDocument();

    fireEvent.click(zoomInBtn);
    expect(screen.getByText("115%")).toBeInTheDocument();

    fireEvent.click(zoomOutBtn);
    expect(screen.getByText("100%")).toBeInTheDocument();

    fireEvent.click(fitPageBtn);
    expect(screen.getByText("85%")).toBeInTheDocument();

    fireEvent.click(fitWidthBtn);
    expect(screen.getByText("100%")).toBeInTheDocument();

    // Page navigation
    const nextBtn = screen.getByRole("button", { name: "Next page" });
    fireEvent.click(nextBtn);

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining("page_number=2"),
        expect.anything()
      );
    });
  });

  // 19. PdfViewerPanel close button and Escape key close panel
  it("PdfViewerPanel close button and Escape key trigger onClose", () => {
    const handleClose = vi.fn();
    const source = {
      source_filename: "atlas.pdf",
      page_number: 1,
      chunk_id: "c1",
      text: "text",
    };

    globalThis.fetch = vi.fn().mockReturnValue(new Promise(() => {}));

    render(
      <PdfViewerPanel
        source={source}
        onClose={handleClose}
        isDocumentAvailable={true}
      />
    );

    const closeBtn = screen.getByRole("button", { name: "Close PDF viewer" });
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);

    // Test Escape key
    fireEvent.keyDown(window, { key: "Escape" });
    expect(handleClose).toHaveBeenCalledTimes(2);
  });

  // 20. PdfViewerPanel renders unavailable state when document is deleted
  it("PdfViewerPanel renders unavailable message when document is not available", () => {
    const handleClose = vi.fn();
    const source = {
      source_filename: "deleted.pdf",
      page_number: 1,
      chunk_id: "del_p1_c0",
      text: "Old evidence from deleted document.",
    };

    render(
      <PdfViewerPanel
        source={source}
        onClose={handleClose}
        isDocumentAvailable={false}
      />
    );

    expect(screen.getByText("Document Unavailable")).toBeInTheDocument();
    expect(
      screen.getByText("This document is no longer available. It may have been deleted.")
    ).toBeInTheDocument();
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });
});
