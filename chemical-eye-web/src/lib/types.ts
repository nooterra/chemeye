export type DetectionStatus = 'RUNNING' | 'COMPLETE' | 'CLEAR' | 'DETECTED' | 'ERROR';

export type DetectionType =
  | 'methane_amf'
  | 'methane'
  | 'tropomi_hotspot'
  | 'unknown';

export interface Detection {
  id: string;
  granule_ur: string;
  status: DetectionStatus;
   detection_type?: DetectionType | string;
  max_z_score: number;
  timestamp: string;
  lat: number;
  lon: number;
  overlay_url: string;
  bounds: [number, number, number, number];
}
