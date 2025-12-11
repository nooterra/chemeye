"use client";

import React from "react";
import Map from "react-map-gl";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, BitmapLayer } from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import "mapbox-gl/dist/mapbox-gl.css";

import { useStore } from "@/lib/store";
import type { Detection } from "@/lib/types";
import { useDetections } from "@/lib/api";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

type LayerableDetection = Detection & { ageDays?: number; result_json?: any };

function computeAgeDays(ts: string | undefined): number | null {
  if (!ts) return null;
  const t = new Date(ts).getTime();
  if (Number.isNaN(t)) return null;
  return (Date.now() - t) / (1000 * 60 * 60 * 24);
}

export function MapView() {
  const { viewState, setViewState, selectedDetection, setSelectedDetection } = useStore();
  const { detections } = useDetections();
  const data: LayerableDetection[] = Array.isArray(detections)
    ? detections.map((d) => ({ ...d, ageDays: computeAgeDays(d.timestamp) ?? undefined }))
    : [];

  if (!Array.isArray(detections)) {
    return null;
  }

  const amfData = data.filter((d) => d.detection_type === "methane_amf" || d.detection_type === "methane");
  const tropomiData = data.filter((d) => d.detection_type === "tropomi_hotspot");

  const layers = [
    // TROPOMI heatmap "cloud" to avoid just dots
    tropomiData.length > 0 &&
      new HeatmapLayer<LayerableDetection>({
        id: "tropomi-heatmap",
        data: tropomiData,
        getPosition: (d) => [d.lon ?? 0, d.lat ?? 0],
        getWeight: (d) => {
          const ppb = (d as any).result_json?.max_ppb ?? d.max_z_score ?? 1;
          return Math.max(1, Number(ppb) || 1);
        },
        radiusPixels: 80,
        intensity: 1,
        threshold: 0.05,
        colorRange: [
          [255, 245, 190],
          [255, 235, 150],
          [255, 220, 100],
          [255, 200, 40],
          [230, 160, 20],
          [200, 120, 10],
        ],
      }),

    // TROPOMI hotspots: yellow, semi-transparent, large radius
    new ScatterplotLayer<LayerableDetection>({
      id: "tropomi-hotspots",
      data: tropomiData,
      getPosition: (d) => [d.lon ?? 0, d.lat ?? 0],
      getFillColor: (d) =>
        ([255, 200, 0, d.ageDays && d.ageDays > 7 ? 80 : 160] as [number, number, number, number]),
      getRadius: () => 5000,
      radiusMinPixels: 8,
      radiusMaxPixels: 80,
      opacity: 0.4,
      pickable: true,
      onClick: ({ object }) => object && setSelectedDetection(object),
    }),

    // AMF detections: red dots
    new ScatterplotLayer<LayerableDetection>({
      id: "detections-circle",
      data: amfData,
      getPosition: (d) => [d.lon ?? 0, d.lat ?? 0],
      getFillColor: (d) => {
        const base = [255, 50, 50];
        // Dim if stale >7 days
        if (d.ageDays && d.ageDays > 7) {
          return [base[0], base[1], base[2], 120] as [number, number, number, number];
        }
        return [base[0], base[1], base[2], 255] as [number, number, number, number];
      },
      getRadius: (d) => (d.max_z_score > 7 ? 2000 : 1000),
      radiusMinPixels: 5,
      radiusMaxPixels: 50,
      opacity: 0.65,
      pickable: true,
      onClick: ({ object }) => object && setSelectedDetection(object),
    }),

    selectedDetection &&
      selectedDetection.overlay_url &&
      new BitmapLayer({
        id: "plume-overlay",
        image: selectedDetection.overlay_url,
        bounds: selectedDetection.bounds,
        opacity: 0.8,
        transparentColor: [0, 0, 0, 0],
      }),
  ].filter(Boolean);

  return (
    <div className="w-full h-screen absolute top-0 left-0">
      <DeckGL
        controller
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState as any)}
        layers={layers as any}
        getTooltip={({ object }) =>
          object && {
            html: `
              <div style="background: rgba(0,0,0,0.8); padding: 8px; color: white; border-radius: 4px; font-family: sans-serif;">
                <strong>Methane Detected</strong><br/>
                Z-Score: ${object.max_z_score}<br/>
                Lat/Lon: ${object.lat.toFixed(3)}, ${object.lon.toFixed(3)}
              </div>
            `,
          }
        }
      >
        <Map
          mapboxAccessToken={MAPBOX_TOKEN}
          mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        />
      </DeckGL>
    </div>
  );
}
