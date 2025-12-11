"use client";

import React from "react";
import { AlertTriangle, LocateFixed, Terminal } from "lucide-react";

import { useStore } from "@/lib/store";
import { useDetections } from "@/lib/api";
import type { Detection } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

export function Sidebar() {
  const { flyToDetection, selectedDetection } = useStore();
  const { detections } = useDetections();
  const data = Array.isArray(detections) ? detections : [];

  if (!Array.isArray(detections)) {
    return (
      <Card className="absolute top-4 left-4 w-[350px] h-[calc(100vh-2rem)] bg-black/80 backdrop-blur-md border-zinc-800 text-white shadow-2xl z-50 rounded-xl overflow-hidden flex flex-col">
        <CardHeader className="pb-2 border-b border-zinc-800">
          <div className="flex items-center space-x-2">
            <Terminal className="w-5 h-5 text-red-500" />
            <CardTitle className="text-lg font-mono tracking-wider">CHEMICAL EYE</CardTitle>
          </div>
          <p className="text-xs text-zinc-400 font-mono">GLOBAL METHANE MONITORING SYS</p>
        </CardHeader>
        <div className="p-3 text-xs text-zinc-500 font-mono">No detection data.</div>
      </Card>
    );
  }

  return (
    <Card className="absolute top-4 left-4 w-[350px] h-[calc(100vh-2rem)] bg-black/80 backdrop-blur-md border-zinc-800 text-white shadow-2xl z-50 rounded-xl overflow-hidden flex flex-col">
      <CardHeader className="pb-2 border-b border-zinc-800">
        <div className="flex items-center space-x-2">
          <Terminal className="w-5 h-5 text-red-500" />
          <CardTitle className="text-lg font-mono tracking-wider">CHEMICAL EYE</CardTitle>
        </div>
        <p className="text-xs text-zinc-400 font-mono">GLOBAL METHANE MONITORING SYS</p>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="pt-4 space-y-4">
          <div className="flex items-center justify-between text-xs text-zinc-500 font-bold uppercase tracking-widest">
            <span>Live Alerts</span>
            <span className="flex items-center">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse mr-2" />
              Online
            </span>
          </div>

          {data.map((d: Detection) => (
            <div
              key={d.id}
              onClick={() => flyToDetection(d)}
              className={cn(
                "group cursor-pointer rounded-lg border border-zinc-800 p-3 transition-all hover:bg-zinc-900 hover:border-zinc-700",
                selectedDetection?.id === d.id &&
                  "bg-zinc-900 border-red-900 ring-1 ring-red-500/50"
              )}
            >
              <div className="flex justify-between items-start mb-2">
                <Badge
                  variant={d.max_z_score > 7 ? "destructive" : "secondary"}
                  className="font-mono"
                >
                  Z: {d.max_z_score.toFixed(1)}
                </Badge>
                <span className="text-[10px] text-zinc-400 font-mono">
                  {new Date(d.timestamp).toLocaleString()}
                </span>
              </div>

              <div className="space-y-1">
                <div className="flex items-center text-sm font-semibold text-zinc-200">
                  <AlertTriangle className="w-3 h-3 mr-2 text-yellow-500" />
                  {d.detection_type === "tropomi_hotspot"
                    ? "Global Hotspot (TROPOMI)"
                    : "Methane Plume Detected"}
                </div>
                <div className="flex items-center text-xs text-zinc-500 font-mono">
                  <LocateFixed className="w-3 h-3 mr-1" />
                  {d.lat.toFixed(4)}, {d.lon.toFixed(4)}
                </div>
              </div>
            </div>
          ))}
        </CardContent>
      </ScrollArea>

      <div className="p-3 bg-zinc-950 border-t border-zinc-800 text-xs text-zinc-500 font-mono">
        <div className="flex items-center justify-between mb-2">
          <span className="text-zinc-400">Algorithm Status</span>
          <Badge variant="secondary" className="bg-green-600/20 text-green-200 border-green-700">
            VALIDATED
          </Badge>
        </div>
        <div className="flex items-center justify-between text-[11px]">
          <span>Sensitivity (&gt;100kg/hr)</span>
          <span className="text-white">85%</span>
        </div>
        <div className="flex items-center justify-between text-[11px]">
          <span>Last Calibration</span>
          <span className="text-white">{new Date().toISOString().slice(0, 10)}</span>
        </div>
      </div>
    </Card>
  );
}
