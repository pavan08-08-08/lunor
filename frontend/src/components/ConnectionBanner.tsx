import React from "react";

interface ConnectionBannerProps {
  isBackendReachable: boolean | null;
}

export const ConnectionBanner: React.FC<ConnectionBannerProps> = ({ isBackendReachable }) => {
  if (isBackendReachable !== false) {
    return null;
  }

  return (
    <div
      role="alert"
      className="bg-amber-50 border-b border-amber-200 px-4 py-2 text-xs text-amber-800 flex items-center justify-between z-30"
    >
      <div className="flex items-center space-x-2">
        <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
        <span className="font-medium">Backend Disconnected</span>
        <span className="hidden sm:inline text-amber-700">
          — Unable to reach API server. Please ensure the backend is running.
        </span>
      </div>
    </div>
  );
};
