import React from "react";
import type { Source } from "../types/api";

interface SourceListProps {
  sources?: Source[];
}

export const SourceList: React.FC<SourceListProps> = ({ sources }) => {
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
        {sources.map((src, index) => (
          <li
            key={`${src.source_filename}-${src.page_number}-${index}`}
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
        ))}
      </ul>
    </div>
  );
};
