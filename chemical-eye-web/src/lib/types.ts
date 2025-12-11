export type DetectionStatus = 'RUNNING' | 'COMPLETE' | 'CLEAR' | 'DETECTED' | 'ERROR';

export interface Detection {
  id: string;
  granule_ur: string;
  status: DetectionStatus;
  max_z_score: number;
  timestamp: string;
  lat: number;
  lon: number;
  overlay_url: string;
  bounds: [number, number, number, number];
}

