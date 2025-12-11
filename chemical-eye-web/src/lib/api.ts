import useSWR from 'swr';
import type { Detection } from './types';

const API_BASE = process.env.NEXT_PUBLIC_CHEMEYE_API_BASE;

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export function useDetections() {
  const { data, error, isLoading } = useSWR<Detection[]>(
    API_BASE ? `${API_BASE}/v1/detections/recent` : null,
    fetcher,
    { refreshInterval: 10_000 },
  );

  return {
    detections: data || [],
    isLoading,
    isError: error,
  };
}

