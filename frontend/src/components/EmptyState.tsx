import React from "react";

interface EmptyStateProps {
  hasDocuments: boolean;
  onSelectPrompt?: (prompt: string) => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  hasDocuments,
  onSelectPrompt,
}) => {
  const examplePrompts = [
    "What are the main findings of the paper?",
    "Summarize the proposed methodology.",
    "What experimental results or metrics were reported?",
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-lg mx-auto">
      <div className="w-12 h-12 rounded-2xl bg-brand-50 border border-brand-100 flex items-center justify-center text-xl mb-4 shadow-sm">
        🌙
      </div>
      <h2 className="text-lg font-semibold text-stone-900 tracking-tight mb-1">
        Lunor Knowledge Assistant
      </h2>
      <p className="text-sm text-stone-500 mb-6 max-w-sm">
        {hasDocuments
          ? "Ask questions grounded directly in your uploaded PDF documents."
          : "Upload a PDF in the sidebar to start asking grounded questions."}
      </p>

      {hasDocuments && onSelectPrompt && (
        <div className="w-full space-y-2">
          <p className="text-[11px] font-medium uppercase tracking-wider text-stone-400 mb-2">
            Suggested questions
          </p>
          <div className="grid gap-2 text-left">
            {examplePrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => onSelectPrompt(prompt)}
                className="text-xs py-2.5 px-3.5 rounded-lg bg-white border border-stone-200 text-stone-700 hover:border-brand-300 hover:bg-brand-50/30 transition-all text-left flex items-center justify-between group shadow-2xs"
              >
                <span>{prompt}</span>
                <span className="text-stone-300 group-hover:text-brand-500 transition-colors">
                  →
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
