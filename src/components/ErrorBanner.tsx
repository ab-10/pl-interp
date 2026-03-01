"use client";

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-red-500/20 bg-red-950/40 px-4 py-2.5 text-[13px] text-red-300">
      <span>{message}</span>
      <button
        onClick={onDismiss}
        className="ml-4 text-red-400/60 hover:text-red-300 transition-colors"
        aria-label="Dismiss error"
      >
        &times;
      </button>
    </div>
  );
}
