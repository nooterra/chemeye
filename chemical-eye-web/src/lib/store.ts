import { FlyToInterpolator } from '@deck.gl/core';
import { create } from 'zustand';
import type { Detection } from './types';

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
  transitionDuration?: number;
  transitionInterpolator?: any;
}

interface DashboardState {
  selectedDetection: Detection | null;
  setSelectedDetection: (detection: Detection | null) => void;
  viewState: ViewState;
  setViewState: (view: Partial<ViewState>) => void;
  flyToDetection: (detection: Detection) => void;
}

export const useStore = create<DashboardState>((set) => ({
  selectedDetection: null,
  viewState: {
    longitude: -103.5,
    latitude: 31.5,
    zoom: 6,
    pitch: 0,
    bearing: 0,
  },
  setSelectedDetection: (d) => set({ selectedDetection: d }),
  setViewState: (v) =>
    set((state) => ({
      viewState: { ...state.viewState, ...v },
    })),
  flyToDetection: (d) =>
    set({
      selectedDetection: d,
      viewState: {
        longitude: d.lon,
        latitude: d.lat,
        zoom: 12,
        pitch: 45,
        bearing: 0,
        transitionDuration: 2000,
        transitionInterpolator: new FlyToInterpolator(),
      },
    }),
}));

