"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import type { PlotMouseEvent } from "plotly.js";
import { FeatureMapPoint } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const VARIANT_COLORS: Record<string, string> = {
  typed: "#22c55e",
  control_flow: "#3b82f6",
  error_handling: "#ef4444",
  decomposition: "#f59e0b",
  baseline: "#71717a",
  invariants: "#a855f7",
  functional_style: "#06b6d4",
  recursion: "#ec4899",
};

interface FeatureMapProps {
  points: FeatureMapPoint[];
  selectedIds: Set<number>;
  onSelect: (id: number) => void;
}

/** Interactive 2D UMAP scatter of SAE features. Click to add as steering slider. */
export default function FeatureMap({ points, selectedIds, onSelect }: FeatureMapProps) {
  const { traces, selectedTrace } = useMemo(() => {
    const groups: Record<string, FeatureMapPoint[]> = {};
    const selected: FeatureMapPoint[] = [];

    for (const p of points) {
      if (selectedIds.has(p.id)) {
        selected.push(p);
      }
      const variant = p.primary_variant ?? "unknown";
      if (!groups[variant]) groups[variant] = [];
      groups[variant].push(p);
    }

    const traces = Object.entries(groups).map(([variant, pts]) => ({
      x: pts.map((p) => p.x),
      y: pts.map((p) => p.y),
      text: pts.map((p) => `#${p.id}${p.label ? `: ${p.label}` : ""}`),
      customdata: pts.map((p) => p.id),
      type: "scattergl" as const,
      mode: "markers" as const,
      name: variant,
      marker: {
        size: 4,
        color: VARIANT_COLORS[variant] ?? "#71717a",
        opacity: 0.6,
      },
      hoverinfo: "text" as const,
    }));

    const selectedTrace =
      selected.length > 0
        ? {
            x: selected.map((p) => p.x),
            y: selected.map((p) => p.y),
            text: selected.map((p) => `#${p.id}: ${p.label ?? "selected"}`),
            customdata: selected.map((p) => p.id),
            type: "scattergl" as const,
            mode: "markers" as const,
            name: "Selected",
            marker: {
              size: 12,
              color: "#fbbf24",
              symbol: "diamond" as const,
              opacity: 1,
              line: { color: "#ffffff", width: 1 },
            },
            hoverinfo: "text" as const,
          }
        : null;

    return { traces, selectedTrace };
  }, [points, selectedIds]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const allTraces = (selectedTrace ? [...traces, selectedTrace] : traces) as any[];

  return (
    <div className="w-full h-full min-h-[400px]">
      <Plot
        data={allTraces}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "#18181b",
          font: { color: "#a1a1aa", size: 10 },
          margin: { t: 10, r: 10, b: 30, l: 30 },
          xaxis: {
            title: { text: "UMAP 1" },
            gridcolor: "#27272a",
            zerolinecolor: "#3f3f46",
          },
          yaxis: {
            title: { text: "UMAP 2" },
            gridcolor: "#27272a",
            zerolinecolor: "#3f3f46",
          },
          legend: {
            orientation: "h",
            y: -0.15,
            font: { size: 10 },
          },
          dragmode: "pan",
        }}
        config={{
          displayModeBar: true,
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
          displaylogo: false,
          scrollZoom: true,
        }}
        style={{ width: "100%", height: "100%" }}
        onClick={(event: Readonly<PlotMouseEvent>) => {
          const point = event.points?.[0];
          if (point?.customdata != null) {
            onSelect(Number(point.customdata));
          }
        }}
      />
    </div>
  );
}
