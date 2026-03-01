"use client";

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-[13px] text-red-700">
      <span>{message}</span>
      <button
        onClick={onDismiss}
        className="ml-4 text-red-400 hover:text-red-600 transition-colors"
        aria-label="Dismiss error"
      >
        &times;
      </button>
    </div>
  );
}
