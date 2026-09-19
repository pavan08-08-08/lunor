import { useEffect, useState } from "react";
import { ConnectionBanner } from "./components/ConnectionBanner";
import { ChatWindow } from "./components/ChatWindow";
import { PdfViewerPanel } from "./components/PdfViewerPanel";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";
import { getHealth } from "./services/api";
import type { Source } from "./types/api";

export function App() {
  const [isBackendReachable, setIsBackendReachable] = useState<boolean | null>(null);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState<boolean>(false);
  const [activeSource, setActiveSource] = useState<Source | null>(null);

  const {
    documents,
    isLoadingDocuments,
    uploadState,
    uploadError,
    lastUpload,
    upload,
    deleteDoc,
    deletingFilename,
  } = useDocuments();

  const { messages, isSending, send } = useChat();

  useEffect(() => {
    let isMounted = true;
    getHealth()
      .then(() => {
        if (isMounted) setIsBackendReachable(true);
      })
      .catch(() => {
        if (isMounted) setIsBackendReachable(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="h-full flex flex-col bg-stone-50 text-stone-900">
      <ConnectionBanner isBackendReachable={isBackendReachable} />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          documents={documents}
          isLoadingDocuments={isLoadingDocuments}
          uploadState={uploadState}
          uploadError={uploadError}
          lastUpload={lastUpload}
          onUpload={upload}
          onDelete={deleteDoc}
          deletingFilename={deletingFilename}
          isOpen={isMobileSidebarOpen}
          onClose={() => setIsMobileSidebarOpen(false)}
          disabled={isBackendReachable === false}
        />

        <ChatWindow
          messages={messages}
          isSending={isSending}
          onSend={send}
          hasDocuments={documents.length > 0}
          isBackendReachable={isBackendReachable}
          onOpenMobileSidebar={() => setIsMobileSidebarOpen(true)}
          onSelectSource={setActiveSource}
        />

        {activeSource && (
          <PdfViewerPanel
            source={activeSource}
            onClose={() => setActiveSource(null)}
            isDocumentAvailable={documents.some(
              (d) => d.filename === activeSource.source_filename
            )}
          />
        )}
      </div>
    </div>
  );
}

export default App;
