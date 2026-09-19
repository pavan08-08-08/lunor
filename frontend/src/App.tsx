import { useEffect, useState } from "react";
import { ConnectionBanner } from "./components/ConnectionBanner";
import { ChatWindow } from "./components/ChatWindow";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";
import { getHealth } from "./services/api";

export function App() {
  const [isBackendReachable, setIsBackendReachable] = useState<boolean | null>(null);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState<boolean>(false);

  const {
    documents,
    isLoadingDocuments,
    uploadState,
    uploadError,
    lastUpload,
    upload,
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
        />
      </div>
    </div>
  );
}

export default App;
