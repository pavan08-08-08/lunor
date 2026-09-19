import React from "react";
import type { Source } from "../types/api";

interface SourceListProps {
  sources?: Source[];
  onSelectSource?: (source: Source) => void;
}

export const SourceList: React.FC<SourceListProps> = ({ sources, onSelectSource }) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 pt-3 border-t border-stone-200/60">
      <div className="text-[11px] font-semibold text-stone-500 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
        <span>Sources</span>
        <span className="text-stone-400 font-normal">({sources.length})</span>
      </div>
      <ul className="space-y-1">
        {sources.map((src, index) => {
          const key = src.chunk_id || `${src.source_filename}-${src.page_number}-${index}`;
          const label = `View evidence in ${src.source_filename}, page ${src.page_number}`;

          if (onSelectSource) {
            return (
              <li key={key}>
                <button
                  type="button"
                  aria-label={label}
                  onClick={() => onSelectSource(src)}
                  className="w-full flex items-center justify-between text-xs py-1.5 px-2 rounded bg-stone-50 hover:bg-stone-100 hover:border-brand-300 border border-stone-200/60 text-stone-700 transition-colors cursor-pointer text-left group"
                >
                  <div className="flex items-center gap-1.5 min-w-0 pr-2">
                    <span className="text-xs shrink-0 group-hover:scale-110 transition-transform" aria-hidden="true">
                      📄
                    </span>
                    <span className="truncate font-medium text-stone-700 group-hover:text-brand-600 transition-colors" title={src.source_filename}>
                      {src.source_filename}
                    </span>
                  </div>
                  <span className="shrink-0 text-stone-500 group-hover:text-brand-600 font-mono text-[11px] flex items-center gap-1">
                    Page {src.page_number}
                    <svg className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </span>
                </button>
              </li>
            );
          }

          return (
            <li
              key={key}
              className="flex items-center justify-between text-xs py-1 px-2 rounded bg-stone-50 border border-stone-200/50 text-stone-700"
            >
              <div className="flex items-center gap-1.5 min-w-0 pr-2">
                <span className="text-xs shrink-0" aria-hidden="true">
                  📄
                </span>
                <span className="truncate font-medium" title={src.source_filename}>
                  {src.source_filename}
                </span>
              </div>
              <span className="shrink-0 text-stone-500 font-mono text-[11px]">
                Page {src.page_number}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
};
