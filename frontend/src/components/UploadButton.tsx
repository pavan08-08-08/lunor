import React, { useRef } from "react";
import type { UploadResponse } from "../types/api";

interface UploadButtonProps {
  onUpload: (file: File) => Promise<boolean>;
  uploadState: "idle" | "uploading" | "error";
  uploadError?: string;
  lastUpload?: UploadResponse;
  disabled?: boolean;
}

export const UploadButton: React.FC<UploadButtonProps> = ({
  onUpload,
  uploadState,
  uploadError,
  lastUpload,
  disabled = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    await onUpload(file);
    // Reset file input value so user can upload the same file again if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isBusy = uploadState === "uploading";
  const isDisabled = disabled || isBusy;

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="sr-only"
        id="pdf-upload-input"
        onChange={handleFileChange}
        disabled={isDisabled}
      />
      <label
        htmlFor="pdf-upload-input"
        className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium transition-colors border shadow-sm ${
          isDisabled
            ? "bg-stone-100 text-stone-400 border-stone-200 cursor-not-allowed"
            : "bg-brand-600 hover:bg-brand-700 text-white border-brand-600 cursor-pointer focus-within:ring-2 focus-within:ring-brand-500 focus-within:ring-offset-2"
        }`}
      >
        {isBusy ? (
          <>
            <svg
              className="animate-spin h-4 w-4 text-white"
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
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            <span>Uploading...</span>
          </>
        ) : (
          <>
            <span className="text-base leading-none font-semibold">+</span>
            <span>Upload PDF</span>
          </>
        )}
      </label>

      {/* Success notification */}
      {lastUpload && uploadState === "idle" && (
        <p className="mt-2 text-xs text-emerald-700 font-medium flex items-center gap-1">
          <span>✓</span>
          <span className="truncate">{lastUpload.filename} uploaded</span>
        </p>
      )}

      {/* Error message */}
      {uploadState === "error" && uploadError && (
        <p className="mt-2 text-xs text-rose-600 font-medium break-words">
          {uploadError}
        </p>
      )}
    </div>
  );
};
