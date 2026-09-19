import { useCallback, useEffect, useState } from "react";
import { deleteDocument, getDocuments, uploadDocument } from "../services/api";
import type { Document, UploadResponse } from "../types/api";

export type UploadState = "idle" | "uploading" | "error";

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState<boolean>(true);
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [uploadError, setUploadError] = useState<string | undefined>(undefined);
  const [lastUpload, setLastUpload] = useState<UploadResponse | undefined>(undefined);
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    setIsLoadingDocuments(true);
    try {
      const data = await getDocuments();
      setDocuments(data.documents || []);
    } catch {
      // Keep existing documents on network error
    } finally {
      setIsLoadingDocuments(false);
    }
  }, []);

  const upload = useCallback(
    async (file: File): Promise<boolean> => {
      setUploadState("uploading");
      setUploadError(undefined);
      setLastUpload(undefined);

      try {
        const response = await uploadDocument(file);
        setLastUpload(response);
        setUploadState("idle");
        await refreshDocuments();
        return true;
      } catch (err: unknown) {
        setUploadState("error");
        const msg = err instanceof Error ? err.message : "Failed to upload document";
        setUploadError(msg);
        return false;
      }
    },
    [refreshDocuments]
  );

  const deleteDoc = useCallback(
    async (filename: string): Promise<boolean> => {
      if (deletingFilename) return false;
      setDeletingFilename(filename);
      try {
        await deleteDocument(filename);
        await refreshDocuments();
        return true;
      } catch {
        return false;
      } finally {
        setDeletingFilename(null);
      }
    },
    [deletingFilename, refreshDocuments]
  );

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  return {
    documents,
    isLoadingDocuments,
    uploadState,
    uploadError,
    lastUpload,
    deletingFilename,
    refreshDocuments,
    upload,
    deleteDoc,
  };
}
