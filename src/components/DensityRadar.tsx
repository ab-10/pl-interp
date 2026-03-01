"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";

const PROPERTY_LABELS: Record<string, string> = {
  type_annotations: "Type Hints",
  error_handling: "Error Handling",
  control_flow: "Control Flow",
  decomposition: "Decomposition",
  functional_style: "Functional",
  recursion: "Recursion",
  verbose_documentation: "Documentation",
};

interface DensityRadarProps {
  baselineDensity: Record<string, number>;
  steeredDensity: Record<string, number>;
}

/** Radar/spider chart comparing property densities between baseline and steered code. */
export default function DensityRadar({ baselineDensity, steeredDensity }: DensityRadarProps) {
  const properties = Object.keys(PROPERTY_LABELS);

  // Check if baseline and steered are identical (no steering applied)
  const isIdentical = properties.every(
    (p) => Math.abs((baselineDensity[p] ?? 0) - (steeredDensity[p] ?? 0)) < 0.001
  );

  const data = properties.map((prop) => ({
    property: PROPERTY_LABELS[prop] ?? prop,
    baseline: baselineDensity[prop] ?? 0,
    steered: steeredDensity[prop] ?? 0,
  }));

  return (
    <div className="w-full" style={{ height: 320 }}>
      {isIdentical && (
        <p className="text-center text-xs text-zinc-400 mb-2">
          No steering applied — baseline and steered densities are identical
        </p>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#e4e4e7" />
          <PolarAngleAxis
            dataKey="property"
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Radar
            name="Baseline"
            dataKey="baseline"
            stroke="#a1a1aa"
            fill="#a1a1aa"
            fillOpacity={0.1}
            strokeWidth={1.5}
          />
          <Radar
            name="Steered"
            dataKey="steered"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.15}
            strokeWidth={2}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#71717a" }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
