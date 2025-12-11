import { MapView } from "@/components/dashboard/map-view";
import { Sidebar } from "@/components/dashboard/sidebar";

export default function Home() {
  return (
    <main className="relative w-full h-screen overflow-hidden bg-black">
      <MapView />
      <Sidebar />
      <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-black/80 to-transparent pointer-events-none z-10" />
    </main>
  );
}
