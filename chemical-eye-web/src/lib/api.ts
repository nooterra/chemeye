import useSWR from 'swr';
import type { Detection } from './types';

const API_BASE = process.env.NEXT_PUBLIC_CHEMEYE_API_BASE;

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json();
};

export function useDetections() {
  const { data, error, isLoading } = useSWR<Detection[]>(
    API_BASE ? `${API_BASE}/v1/detections/recent` : null,
    fetcher,
    { refreshInterval: 10_000 },
  );

  const safeData = !error && Array.isArray(data) ? data : [];

  return {
    detections: safeData,
    isLoading,
    isError: error,
  };
}
