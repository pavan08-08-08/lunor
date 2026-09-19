import React from "react";
import type { Document } from "../types/api";

interface DocumentListProps {
  documents: Document[];
  isLoading: boolean;
  onDelete?: (filename: string) => Promise<boolean>;
  deletingFilename?: string | null;
  disabled?: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  isLoading,
  onDelete,
  deletingFilename,
  disabled = false,
}) => {
  if (isLoading && documents.length === 0) {
    return (
      <div className="py-6 text-center text-xs text-stone-400 animate-pulse">
        Loading documents...
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="py-6 px-2 text-center text-stone-400">
        <p className="text-xs font-medium text-stone-500 mb-1">No documents yet</p>
        <p className="text-[11px] leading-relaxed">
          Upload a PDF to start asking questions.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-1 overflow-y-auto max-h-full">
      <div className="text-[11px] font-medium tracking-wider text-stone-400 uppercase px-2 mb-2">
        Indexed Files ({documents.length})
      </div>
      <ul className="space-y-1">
        {documents.map((doc, index) => {
          const isDeleting = deletingFilename === doc.filename;
          return (
            <li
              key={`${doc.filename}-${index}`}
              className="flex items-center justify-between gap-2 px-2.5 py-2 rounded-md text-xs text-stone-700 hover:bg-stone-100 transition-colors group"
            >
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <span className="text-sm shrink-0" aria-hidden="true">
                  📄
                </span>
                <span className="truncate font-medium" title={doc.filename}>
                  {doc.filename}
                </span>
              </div>
              {onDelete && (
                <button
                  type="button"
                  aria-label={`Delete ${doc.filename}`}
                  disabled={disabled || isDeleting}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(doc.filename);
                  }}
                  className="p-1 rounded text-stone-400 hover:text-rose-600 hover:bg-rose-50 transition-colors shrink-0 disabled:opacity-40 disabled:hover:text-stone-400 disabled:hover:bg-transparent"
                  title={`Delete ${doc.filename}`}
                >
                  {isDeleting ? (
                    <svg
                      className="w-3.5 h-3.5 animate-spin text-stone-400"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v8H4z"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  )}
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
};
