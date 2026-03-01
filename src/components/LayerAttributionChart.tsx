"use client";

import { LayerAttribution } from "@/lib/types";

interface LayerAttributionChartProps {
  attribution: LayerAttribution[];
  featureId: number;
  featureActivation: number;
  token: string;
}

const CHART_WIDTH = 500;
const BAR_HEIGHT = 18;
const LABEL_WIDTH = 60;
const RIGHT_MARGIN = 50;
const TOP_MARGIN = 10;

export default function LayerAttributionChart({
  attribution,
  featureId,
  featureActivation,
  token,
}: LayerAttributionChartProps) {
  const numLayers = attribution.length;
  const chartHeight = numLayers * (BAR_HEIGHT + 4) + TOP_MARGIN + 30;

  // Compute scale: find max absolute value across all contributions
  const maxAbsVal = Math.max(
    ...attribution.flatMap((a) => [Math.abs(a.attn), Math.abs(a.mlp)]),
    0.01 // avoid division by zero
  );
  const barAreaWidth = (CHART_WIDTH - LABEL_WIDTH - RIGHT_MARGIN) / 2;
  const centerX = LABEL_WIDTH + barAreaWidth;
  const scale = barAreaWidth / maxAbsVal;

  // Cumulative sum
  const cumulativeValues: number[] = [];
  let cumSum = 0;
  for (const a of attribution) {
    cumSum += a.attn + a.mlp;
    cumulativeValues.push(cumSum);
  }
  const finalCumulative = cumSum;

  // Scale for cumulative line (maps to full bar area)
  const cumMax = Math.max(...cumulativeValues.map(Math.abs), Math.abs(finalCumulative), 0.01);
  const cumScale = barAreaWidth / cumMax;

  // Build cumulative line points
  const cumPoints = cumulativeValues.map((val, i) => {
    const x = centerX + val * cumScale;
    const y = TOP_MARGIN + i * (BAR_HEIGHT + 4) + BAR_HEIGHT / 2;
    return `${x},${y}`;
  });

  return (
    <div data-testid="layer-attribution">
      <h3 className="text-sm font-medium text-zinc-300 mb-2">
        Layer Attribution for Feature {featureId} at &ldquo;{token}&rdquo;
      </h3>

      <svg
        width={CHART_WIDTH}
        height={chartHeight}
        className="font-mono"
      >
        {/* Zero line */}
        <line
          data-testid="zero-line"
          x1={centerX}
          y1={0}
          x2={centerX}
          y2={chartHeight - 20}
          stroke="#71717a"
          strokeWidth={1}
          strokeDasharray="2,2"
        />

        {attribution.map((a, i) => {
          const y = TOP_MARGIN + i * (BAR_HEIGHT + 4);
          const totalContrib = a.attn + a.mlp;
          const direction = totalContrib >= 0 ? "positive" : "negative";

          // Attention bar
          const attnWidth = Math.abs(a.attn) * scale;
          const attnX = a.attn >= 0 ? centerX : centerX - attnWidth;

          // MLP bar — stacked after attention in the same direction
          const mlpWidth = Math.abs(a.mlp) * scale;
          let mlpX: number;
          if (a.attn >= 0 && a.mlp >= 0) {
            mlpX = centerX + attnWidth;
          } else if (a.attn < 0 && a.mlp < 0) {
            mlpX = centerX - attnWidth - mlpWidth;
          } else if (a.mlp >= 0) {
            mlpX = centerX;
          } else {
            mlpX = centerX - mlpWidth;
          }

          const isMinimal = Math.abs(totalContrib) < maxAbsVal * 0.02;

          return (
            <g key={i} data-testid="layer-bar" data-direction={direction}>
              {/* Layer label */}
              <text
                x={LABEL_WIDTH - 8}
                y={y + BAR_HEIGHT / 2 + 4}
                fill={isMinimal ? "#52525b" : "#a1a1aa"}
                fontSize={11}
                textAnchor="end"
              >
                L{a.layer}
              </text>

              {/* Attention bar */}
              {attnWidth > 0.5 && (
                <rect
                  x={attnX}
                  y={y}
                  width={attnWidth}
                  height={BAR_HEIGHT}
                  fill="#3b82f6"
                  opacity={isMinimal ? 0.3 : 0.9}
                  rx={2}
                />
              )}

              {/* MLP bar */}
              {mlpWidth > 0.5 && (
                <rect
                  x={mlpX}
                  y={y}
                  width={mlpWidth}
                  height={BAR_HEIGHT}
                  fill="#f59e0b"
                  opacity={isMinimal ? 0.3 : 0.9}
                  rx={2}
                />
              )}
            </g>
          );
        })}

        {/* Cumulative line */}
        <polyline
          data-testid="cumulative-line"
          data-final-value={finalCumulative.toFixed(4)}
          points={cumPoints.join(" ")}
          fill="none"
          stroke="#f43f5e"
          strokeWidth={2}
          strokeLinejoin="round"
        />

        {/* Cumulative endpoint label */}
        {cumulativeValues.length > 0 && (
          <text
            x={centerX + finalCumulative * cumScale + 4}
            y={TOP_MARGIN + (numLayers - 1) * (BAR_HEIGHT + 4) + BAR_HEIGHT / 2 + 4}
            fill="#f43f5e"
            fontSize={11}
          >
            {finalCumulative.toFixed(1)}
          </text>
        )}

        {/* Legend */}
        <g transform={`translate(${LABEL_WIDTH}, ${chartHeight - 16})`}>
          <rect x={0} y={0} width={10} height={10} fill="#3b82f6" rx={2} />
          <text x={14} y={9} fill="#a1a1aa" fontSize={10}>Attn</text>
          <rect x={50} y={0} width={10} height={10} fill="#f59e0b" rx={2} />
          <text x={64} y={9} fill="#a1a1aa" fontSize={10}>MLP</text>
          <line x1={100} y1={5} x2={120} y2={5} stroke="#f43f5e" strokeWidth={2} />
          <text x={124} y={9} fill="#a1a1aa" fontSize={10}>Cumulative</text>
        </g>
      </svg>
    </div>
  );
}
