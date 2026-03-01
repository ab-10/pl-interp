"use client";

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-red-50 px-4 py-3 text-sm text-red-800">
      <span>{message}</span>
      <button
        onClick={onDismiss}
        className="ml-4 text-red-600 hover:text-red-800"
        aria-label="Dismiss error"
      >
        &times;
      </button>
    </div>
  );
}
