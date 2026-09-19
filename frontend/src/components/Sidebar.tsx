import React from "react";
import type { Document, UploadResponse } from "../types/api";
import { DocumentList } from "./DocumentList";
import { UploadButton } from "./UploadButton";

interface SidebarProps {
  documents: Document[];
  isLoadingDocuments: boolean;
  uploadState: "idle" | "uploading" | "error";
  uploadError?: string;
  lastUpload?: UploadResponse;
  onUpload: (file: File) => Promise<boolean>;
  onDelete?: (filename: string) => Promise<boolean>;
  deletingFilename?: string | null;
  isOpen: boolean;
  onClose: () => void;
  disabled?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  documents,
  isLoadingDocuments,
  uploadState,
  uploadError,
  lastUpload,
  onUpload,
  onDelete,
  deletingFilename,
  isOpen,
  onClose,
  disabled = false,
}) => {
  const handleUploadWithMobileClose = async (file: File) => {
    const success = await onUpload(file);
    if (success) {
      // Close mobile drawer on successful upload
      onClose();
    }
    return success;
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-stone-900/30 backdrop-blur-sm z-40 md:hidden transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-50 w-72 md:w-64 lg:w-72 bg-white border-r border-stone-200 flex flex-col transition-transform duration-200 ease-in-out shrink-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
        aria-label="Document Management Sidebar"
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-stone-100 flex items-center justify-between">
          <h2 className="text-xs font-semibold tracking-wider text-stone-500 uppercase">
            Documents
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="md:hidden p-1 text-stone-400 hover:text-stone-700 rounded"
            aria-label="Close sidebar"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Upload section */}
        <div className="p-4 border-b border-stone-100">
          <UploadButton
            onUpload={handleUploadWithMobileClose}
            uploadState={uploadState}
            uploadError={uploadError}
            lastUpload={lastUpload}
            disabled={disabled}
          />
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto p-3">
          <DocumentList
            documents={documents}
            isLoading={isLoadingDocuments}
            onDelete={onDelete}
            deletingFilename={deletingFilename}
            disabled={disabled}
          />
        </div>
      </aside>
    </>
  );
};
