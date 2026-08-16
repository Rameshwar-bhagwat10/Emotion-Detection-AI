"use client";

export default function Error({
  error: _error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold">Something went wrong!</h2>
      <button onClick={() => reset()} className="mt-4 px-4 py-2 bg-primary text-white rounded">
        Try again
      </button>
    </div>
  );
}
