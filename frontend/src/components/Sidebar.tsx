import React from "react";
import type { Conversation, Document, UploadResponse } from "../types/api";
import { ConversationList } from "./ConversationList";
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
  conversations?: Conversation[];
  activeConversationId?: string;
  onSelectConversation?: (id: string) => void;
  onNewChat?: () => void;
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
  conversations = [],
  activeConversationId = "",
  onSelectConversation,
  onNewChat,
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
        {/* Top bar with New Chat and mobile close */}
        <div className="p-3 border-b border-stone-200 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold tracking-wider text-stone-500 uppercase">
              Workspace
            </span>
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

          <button
            type="button"
            onClick={() => {
              onNewChat?.();
              onClose();
            }}
            disabled={disabled}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-stone-700 bg-stone-50 hover:bg-stone-100 hover:text-stone-900 border border-stone-200 rounded-lg shadow-xs transition-colors cursor-pointer disabled:opacity-50"
            aria-label="New chat"
          >
            <svg
              className="w-3.5 h-3.5 text-stone-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            <span>New Chat</span>
          </button>
        </div>

        {/* Conversations Section */}
        {conversations.length > 0 && (
          <div className="border-b border-stone-200 py-2.5">
            <div className="px-3 pb-1.5 flex items-center justify-between text-[11px] font-semibold tracking-wider text-stone-400 uppercase">
              <span>Chats</span>
              <span className="font-normal text-stone-400">({conversations.length})</span>
            </div>
            <div className="max-h-48 overflow-y-auto px-2">
              <ConversationList
                conversations={conversations}
                activeConversationId={activeConversationId}
                onSelectConversation={(id) => {
                  onSelectConversation?.(id);
                  onClose();
                }}
                disabled={disabled}
              />
            </div>
          </div>
        )}

        {/* Documents Section */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="px-3 pt-3 pb-1 flex items-center justify-between text-[11px] font-semibold tracking-wider text-stone-400 uppercase">
            <span>Documents</span>
            <span className="font-normal text-stone-400">({documents.length})</span>
          </div>

          {/* Upload section */}
          <div className="p-3 border-b border-stone-100">
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
        </div>
      </aside>
    </>
  );
};
