import React from "react";
import type { Document } from "../types/api";

interface DocumentListProps {
  documents: Document[];
  isLoading: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({ documents, isLoading }) => {
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
        {documents.map((doc, index) => (
          <li
            key={`${doc.filename}-${index}`}
            className="flex items-center gap-2 px-2.5 py-2 rounded-md text-xs text-stone-700 hover:bg-stone-100 transition-colors group"
          >
            <span className="text-sm shrink-0" aria-hidden="true">
              📄
            </span>
            <span className="truncate flex-1 font-medium" title={doc.filename}>
              {doc.filename}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
};
