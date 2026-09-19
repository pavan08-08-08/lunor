export interface Document {
  filename: string;
}

export interface DocumentsResponse {
  documents: Document[];
}

export interface HighlightRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PageRenderResponse {
  filename: string;
  page_number: number;
  total_pages: number;
  page_width: number;
  page_height: number;
  svg: string;
  evidence_text: string;
  highlights: HighlightRect[];
}

export interface Source {
  source_filename: string;
  page_number: number;
  chunk_id: string;
  text: string;
  evidence_text?: string;
}

export type ChatMessageStatus = "sending" | "sent" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  hasSufficientContext?: boolean;
  status: ChatMessageStatus;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

export interface ChatRequest {
  query: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  has_sufficient_context: boolean;
}

export interface UploadResponse {
  message: string;
  filename: string;
  pages: number;
  chunks: number;
}

export interface DeleteResponse {
  message: string;
  filename: string;
  remaining_documents: number;
}

export interface HealthResponse {
  status: string;
}
