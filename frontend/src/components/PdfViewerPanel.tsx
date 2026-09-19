import React, { useEffect, useRef, useState } from "react";
import type { PageRenderResponse, Source } from "../types/api";
import { getDocumentPage } from "../services/api";

interface PdfViewerPanelProps {
  source: Source;
  onClose: () => void;
  isDocumentAvailable?: boolean;
}

export const PdfViewerPanel: React.FC<PdfViewerPanelProps> = ({
  source,
  onClose,
  isDocumentAvailable = true,
}) => {
  const [currentPage, setCurrentPage] = useState<number>(source.page_number);
  const [zoomScale, setZoomScale] = useState<number>(1.0);
  const [pageData, setPageData] = useState<PageRenderResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const firstHighlightRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Sync current page when source changes
  useEffect(() => {
    setCurrentPage(source.page_number);
  }, [source.source_filename, source.page_number, source.chunk_id]);

  // Listen for Escape key to close panel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Fetch page SVG and highlight coordinates
  useEffect(() => {
    if (!isDocumentAvailable) {
      setIsLoading(false);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    const chunkIdToPass = currentPage === source.page_number ? source.chunk_id : undefined;
    const evidenceTextToPass = currentPage === source.page_number ? (source.evidence_text || source.text) : undefined;

    getDocumentPage(
      source.source_filename,
      currentPage,
      chunkIdToPass,
      undefined,
      evidenceTextToPass
    )
      .then((data) => {
        if (!isMounted) return;
        setPageData(data);
        setIsLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err.message || "Failed to load document page");
        setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [source.source_filename, source.chunk_id, source.page_number, currentPage, isDocumentAvailable]);

  // Auto-scroll to first highlight when page finishes loading
  useEffect(() => {
    if (!isLoading && pageData && pageData.highlights.length > 0) {
      const timer = setTimeout(() => {
        firstHighlightRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isLoading, pageData]);

  const totalPages = pageData?.total_pages ?? source.page_number;

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage((prev) => prev - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage((prev) => prev + 1);
    }
  };

  const handleZoomIn = () => {
    setZoomScale((prev) => Math.min(2.5, Math.round((prev + 0.15) * 100) / 100));
  };

  const handleZoomOut = () => {
    setZoomScale((prev) => Math.max(0.5, Math.round((prev - 0.15) * 100) / 100));
  };

  const handleFitWidth = () => {
    setZoomScale(1.0);
  };

  const handleFitPage = () => {
    setZoomScale(0.85);
  };

  const handleResetZoom = () => {
    setZoomScale(1.0);
  };

  return (
    <aside
      role="dialog"
      aria-modal="true"
      aria-label={`PDF Evidence Viewer - ${source.source_filename}`}
      className="fixed md:static inset-0 z-50 md:z-20 w-full md:w-[500px] lg:w-[600px] xl:w-[680px] bg-white border-l border-stone-200 flex flex-col h-full shadow-xl md:shadow-none shrink-0 animate-in slide-in-from-right duration-200"
    >
      {/* Panel Header */}
      <div className="h-14 px-4 border-b border-stone-200 flex items-center justify-between bg-white shrink-0">
        <div className="flex items-center gap-2 min-w-0 pr-2">
          <span className="text-sm shrink-0" aria-hidden="true">
            📄
          </span>
          <div className="min-w-0">
            <h2
              className="text-xs font-semibold text-stone-800 truncate max-w-[160px] sm:max-w-[200px]"
              title={source.source_filename}
            >
              {source.source_filename}
            </h2>
            <span className="text-[11px] font-mono text-stone-400">
              Page {currentPage} of {totalPages}
            </span>
          </div>
        </div>

        {/* Page Navigation & Close */}
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="flex items-center border border-stone-200 rounded-md overflow-hidden bg-stone-50">
            <button
              type="button"
              onClick={handlePrevPage}
              disabled={currentPage <= 1 || isLoading}
              aria-label="Previous page"
              title="Previous page"
              className="px-2 py-1 text-xs text-stone-600 hover:text-stone-900 hover:bg-stone-200 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
            >
              ◀
            </button>
            <span className="px-2 py-1 text-[11px] font-mono text-stone-600 bg-white border-x border-stone-200">
              {currentPage} / {totalPages}
            </span>
            <button
              type="button"
              onClick={handleNextPage}
              disabled={currentPage >= totalPages || isLoading}
              aria-label="Next page"
              title="Next page"
              className="px-2 py-1 text-xs text-stone-600 hover:text-stone-900 hover:bg-stone-200 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
            >
              ▶
            </button>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close PDF viewer"
            className="p-1.5 rounded text-stone-400 hover:text-stone-700 hover:bg-stone-100 transition-colors ml-1"
            title="Close (Esc)"
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
      </div>

      {/* Zoom / View Toolbar */}
      <div className="px-4 py-1.5 bg-stone-50/90 border-b border-stone-200 flex items-center justify-between text-xs text-stone-600 shrink-0">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleFitWidth}
            aria-label="Fit to width"
            title="Fit to width"
            className="px-2 py-0.5 rounded text-[11px] font-medium border border-stone-200 bg-white hover:bg-stone-100 transition-colors"
          >
            Fit Width
          </button>
          <button
            type="button"
            onClick={handleFitPage}
            aria-label="Fit to page"
            title="Fit to page"
            className="px-2 py-0.5 rounded text-[11px] font-medium border border-stone-200 bg-white hover:bg-stone-100 transition-colors"
          >
            Fit Page
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleZoomOut}
            disabled={zoomScale <= 0.5}
            aria-label="Zoom out"
            title="Zoom out"
            className="w-6 h-6 flex items-center justify-center rounded border border-stone-200 bg-white hover:bg-stone-100 disabled:opacity-30 transition-colors text-xs font-semibold"
          >
            -
          </button>
          <button
            type="button"
            onClick={handleResetZoom}
            aria-label="Reset zoom"
            title="Reset zoom to 100%"
            className="px-1.5 py-0.5 text-[11px] font-mono text-stone-600 hover:text-stone-900 transition-colors"
          >
            {Math.round(zoomScale * 100)}%
          </button>
          <button
            type="button"
            onClick={handleZoomIn}
            disabled={zoomScale >= 2.5}
            aria-label="Zoom in"
            title="Zoom in"
            className="w-6 h-6 flex items-center justify-center rounded border border-stone-200 bg-white hover:bg-stone-100 disabled:opacity-30 transition-colors text-xs font-semibold"
          >
            +
          </button>
        </div>
      </div>

      {/* Retrieved Evidence Card */}
      <div className="p-3 bg-amber-50/60 border-b border-amber-200/60 shrink-0 max-h-40 overflow-y-auto">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] font-semibold text-amber-900 tracking-wider uppercase flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
            Retrieved Evidence Passage
          </span>
          <span className="text-[10px] font-mono text-amber-800/70">
            Page {source.page_number}
          </span>
        </div>
        <div className="bg-white/80 border border-amber-200/80 rounded p-2 text-xs text-stone-800 leading-relaxed font-sans whitespace-pre-wrap selection:bg-amber-200 shadow-xs">
          {source.evidence_text || source.text}
        </div>
      </div>

      {/* PDF Page SVG Display / States */}
      <div
        ref={scrollContainerRef}
        className="flex-1 bg-stone-100 relative overflow-auto flex flex-col items-center p-4 min-h-0"
      >
        {!isDocumentAvailable ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
            <div className="w-10 h-10 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center text-lg mb-2.5">
              ⚠️
            </div>
            <h3 className="text-xs font-semibold text-stone-800 mb-1">
              Document Unavailable
            </h3>
            <p className="text-[11px] text-stone-500 max-w-xs leading-relaxed">
              This document is no longer available. It may have been deleted.
            </p>
          </div>
        ) : isLoading ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-stone-500">
            <div className="w-7 h-7 border-2 border-stone-300 border-t-brand-600 rounded-full animate-spin mb-3" />
            <p className="text-xs font-medium">Loading page {currentPage}...</p>
          </div>
        ) : error ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-rose-600">
            <p className="text-xs font-medium mb-1">Error rendering page</p>
            <p className="text-[11px] text-stone-500">{error}</p>
          </div>
        ) : pageData ? (
          <div
            data-pdf-svg-container="true"
            className="relative bg-white shadow-lg border border-stone-200 transition-all duration-150 rounded-xs select-text my-auto shrink-0"
            style={{
              width: `${Math.round(zoomScale * 100)}%`,
              maxWidth: "100%",
              aspectRatio: `${pageData.page_width} / ${pageData.page_height}`,
            }}
          >
            <style>{`
              [data-pdf-svg-container] svg {
                width: 100% !important;
                height: 100% !important;
                display: block;
              }
            `}</style>

            {/* In-Memory Crisp Vector SVG Page (Highlight rendered in-backend directly in SVG) */}
            <div
              className="w-full h-full pointer-events-none"
              dangerouslySetInnerHTML={{ __html: pageData.svg }}
            />

            {/* Invisible target for auto-scroll and test assertions */}
            {pageData.highlights.length > 0 && (
              <div
                ref={firstHighlightRef}
                data-testid="evidence-highlight"
                className="absolute pointer-events-none opacity-0"
                style={{
                  left: `${(pageData.highlights[0].x / pageData.page_width) * 100}%`,
                  top: `${(pageData.highlights[0].y / pageData.page_height) * 100}%`,
                  width: `${(pageData.highlights[0].width / pageData.page_width) * 100}%`,
                  height: `${(pageData.highlights[0].height / pageData.page_height) * 100}%`,
                }}
              />
            )}
          </div>
        ) : null}
      </div>
    </aside>
  );
};
