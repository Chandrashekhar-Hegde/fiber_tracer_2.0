import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { Theme } from "../theme";
import type { AnalysisConfig } from "../types";

interface ConfigureProps {
  config: AnalysisConfig;
  onChange: (config: AnalysisConfig) => void;
  theme: Theme;
}

const TOGGLES: { key: keyof AnalysisConfig; label: string }[] = [
  { key: "computeMorphometry", label: "Morphometry" },
  { key: "computeOrientationTensor", label: "Orientation tensor" },
  { key: "computeTracking", label: "Centerline tracking" },
  { key: "computeTda", label: "TDA descriptors" },
  { key: "computeDvc", label: "DVC (digital volume correlation)" },
];

// ponytail: reuses SelectData's hardcoded recent-files list rather than a
// real file browser; matches the existing single-volume picker's scope.
const RECENT_FILES = ["data/gfpa66_center.tif", "data/sample_a.tif", "data/sample_b.tif"];

export function Configure({ config, onChange, theme }: ConfigureProps) {
  const rows = config.computeDvc
    ? [
        ...TOGGLES,
        { key: "dvcReferencePath" as const, label: "DVC reference volume" },
        { key: "dvcDeformedPath" as const, label: "DVC deformed volume" },
      ]
    : TOGGLES;
  const [selectedIdx, setSelectedIdx] = useState(0);
  const clampedIdx = Math.min(selectedIdx, rows.length - 1);

  useInput((input, key) => {
    if (key.upArrow) setSelectedIdx((i) => Math.max(0, i - 1));
    if (key.downArrow) setSelectedIdx((i) => Math.min(rows.length - 1, i + 1));
    if (input === " " || key.return) {
      const row = rows[clampedIdx];
      if (row.key === "dvcReferencePath" || row.key === "dvcDeformedPath") {
        const current = RECENT_FILES.indexOf(config[row.key] as string);
        const next = RECENT_FILES[(current + 1) % RECENT_FILES.length];
        onChange({ ...config, [row.key]: next });
      } else {
        onChange({ ...config, [row.key]: !config[row.key] });
      }
    }
  });

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        2. Configure
      </Text>
      <Text color={theme.foreground}>Fiber diameter: {config.fiberDiameter} µm</Text>
      <Text color={theme.foreground}>Regime: {config.regime}</Text>
      <Text color={theme.foreground}>Method: {config.method}</Text>
      <Text color={theme.foreground}>
        Threshold: {config.thresholdMethod}
        {config.thresholdMethod === "manual" && config.thresholdValue !== undefined
          ? ` (${config.thresholdValue})`
          : ""}
      </Text>
      <Text color={theme.foreground}>Batch size: {config.batchSize}</Text>
      <Box marginTop={1} flexDirection="column">
        {rows.map((row, idx) => {
          const isFilePicker = row.key === "dvcReferencePath" || row.key === "dvcDeformedPath";
          const value = isFilePicker
            ? (config[row.key] as string) || "none"
            : config[row.key]
              ? "on"
              : "off";
          return (
            <Text key={row.key} color={idx === clampedIdx ? theme.highlight : theme.foreground}>
              {idx === clampedIdx ? "▸ " : "  "}
              {row.label}: {value}
            </Text>
          );
        })}
      </Box>
      <Text color={theme.muted}>
        Use ↑ ↓ to select, Space/Enter to toggle or cycle. ← → to navigate steps.
      </Text>
    </Box>
  );
}
