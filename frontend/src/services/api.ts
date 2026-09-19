import type {
  ChatRequest,
  ChatResponse,
  DeleteResponse,
  DocumentsResponse,
  HealthResponse,
  PageRenderResponse,
  UploadResponse,
} from "../types/api";

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(status: number, message: string, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export const rawBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://localhost:8000";
export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, "");

export function getDocumentPdfUrl(filename: string, chunkId?: string, pageNumber?: number): string {
  const encodedName = encodeURIComponent(filename);
  let url = `${API_BASE_URL}/api/documents/${encodedName}/file`;
  const params = new URLSearchParams();
  if (chunkId) {
    params.set("chunk_id", chunkId);
  }
  const queryString = params.toString();
  if (queryString) {
    url += `?${queryString}`;
  }
  if (pageNumber) {
    url += `#page=${pageNumber}`;
  }
  return url;
}

export async function getDocumentPage(
  filename: string,
  pageNumber: number = 1,
  chunkId?: string,
  query?: string,
  evidenceText?: string,
): Promise<PageRenderResponse> {
  const encodedName = encodeURIComponent(filename);
  const params = new URLSearchParams();
  params.set("page_number", String(pageNumber));
  if (chunkId) {
    params.set("chunk_id", chunkId);
  }
  if (query) {
    params.set("query", query);
  }
  if (evidenceText) {
    params.set("evidence_text", evidenceText);
  }
  const res = await fetch(`${API_BASE_URL}/api/documents/${encodedName}/page?${params.toString()}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return handleResponse<PageRenderResponse>(res);
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail: string | undefined;
    try {
      const errorJson = await response.json();
      if (typeof errorJson?.detail === "string") {
        errorDetail = errorJson.detail;
      } else if (Array.isArray(errorJson?.detail)) {
        // FastAPI pydantic validation errors
        errorDetail = errorJson.detail.map((d: { msg?: string }) => d.msg || "").filter(Boolean).join("; ");
      }
    } catch {
      // Non-JSON response body or empty
    }

    if (response.status === 429) {
      const message = errorDetail || "Gemini API quota exhausted. Please try again after the quota resets.";
      throw new ApiError(429, message, errorDetail);
    }

    const message = errorDetail || `Request failed with status ${response.status}`;
    throw new ApiError(response.status, message, errorDetail);
  }

  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return handleResponse<HealthResponse>(res);
}

export async function getDocuments(): Promise<DocumentsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/documents`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return handleResponse<DocumentsResponse>(res);
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/api/documents/upload`, {
    method: "POST",
    headers: { Accept: "application/json" },
    body: formData,
  });
  return handleResponse<UploadResponse>(res);
}

export async function deleteDocument(filename: string): Promise<DeleteResponse> {
  const res = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });
  return handleResponse<DeleteResponse>(res);
}

export async function sendChatMessage(query: string): Promise<ChatResponse> {
  const payload: ChatRequest = { query };
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse<ChatResponse>(res);
}
