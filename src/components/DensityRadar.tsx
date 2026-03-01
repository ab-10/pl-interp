"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
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

  const data = properties.map((prop) => ({
    property: PROPERTY_LABELS[prop] ?? prop,
    baseline: baselineDensity[prop] ?? 0,
    steered: steeredDensity[prop] ?? 0,
  }));

  // Find max for domain scaling
  const maxVal = Math.max(
    ...data.map((d) => Math.max(d.baseline, d.steered)),
    0.1,
  );

  return (
    <div className="w-full" style={{ height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#3f3f46" />
          <PolarAngleAxis
            dataKey="property"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, Math.ceil(maxVal * 10) / 10]}
            tick={{ fill: "#71717a", fontSize: 9 }}
            tickCount={4}
          />
          <Radar
            name="Baseline"
            dataKey="baseline"
            stroke="#71717a"
            fill="#71717a"
            fillOpacity={0.15}
            strokeWidth={1.5}
          />
          <Radar
            name="Steered"
            dataKey="steered"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.2}
            strokeWidth={2}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
