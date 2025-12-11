"use client";

import React from "react";
import Map from "react-map-gl";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, BitmapLayer } from "@deck.gl/layers";
import "mapbox-gl/dist/mapbox-gl.css";

import { useStore } from "@/lib/store";
import type { Detection } from "@/lib/types";
import { useDetections } from "@/lib/api";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

export function MapView() {
  const { viewState, setViewState, selectedDetection, setSelectedDetection } = useStore();
  const { detections } = useDetections();
  const data = Array.isArray(detections) ? detections : [];

  if (!Array.isArray(detections)) {
    return null;
  }

  const layers = [
    new ScatterplotLayer<Detection>({
      id: "detections-circle",
      data,
      getPosition: (d) => [d.lon ?? 0, d.lat ?? 0],
      getFillColor: [255, 50, 50],
      getRadius: (d) => (d.max_z_score > 7 ? 2000 : 1000),
      radiusMinPixels: 5,
      radiusMaxPixels: 50,
      opacity: 0.6,
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
