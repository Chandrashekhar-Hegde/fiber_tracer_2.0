import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { listExperiments } from "../bridge";
import type { Theme } from "../theme";
import type { Experiment } from "../types";

interface ExperimentsProps {
  theme: Theme;
}

function statusIcon(status: Experiment["status"]): string {
  switch (status) {
    case "completed":
      return "✓";
    case "running":
      return "▶";
    case "failed":
      return "✗";
    case "cancelled":
      return "⊘";
    case "pending":
    default:
      return "○";
  }
}

export function Experiments({ theme }: ExperimentsProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listExperiments()
      .then((data) => {
        if (!cancelled) {
          setExperiments(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Experiments</Text>
      {loading && <Text color={theme.muted}>Loading experiments...</Text>}
      {error && <Text color={theme.error}>Error: {error}</Text>}
      {!loading && !error && experiments.length === 0 && (
        <Text color={theme.muted}>No experiments found.</Text>
      )}
      {!loading &&
        experiments.map((e) => (
          <Text key={e.id} color={theme.foreground}>
            {statusIcon(e.status)} {e.name} — model: {e.modelId} ({e.status})
          </Text>
        ))}
      <Text color={theme.muted}>Track hyper-parameter sweeps and compare runs (v3.3+).</Text>
    </Box>
  );
}
