export interface Document {
  filename: string;
}

export interface DocumentsResponse {
  documents: Document[];
}

export interface Source {
  source_filename: string;
  page_number: number;
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
