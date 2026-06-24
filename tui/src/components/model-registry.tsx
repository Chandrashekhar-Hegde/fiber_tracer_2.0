import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { listModels } from "../bridge";
import type { Theme } from "../theme";
import type { Model } from "../types";

interface ModelRegistryProps {
  theme: Theme;
}

export function ModelRegistry({ theme }: ModelRegistryProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listModels()
      .then((data) => {
        if (!cancelled) {
          setModels(data);
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
      <Text bold color={theme.accent}>Model Registry</Text>
      {loading && <Text color={theme.muted}>Loading models...</Text>}
      {error && <Text color={theme.error}>Error: {error}</Text>}
      {!loading && !error && models.length === 0 && (
        <Text color={theme.muted}>No models found.</Text>
      )}
      {!loading &&
        models.map((m) => (
          <Text key={m.id} color={theme.foreground}>
            {m.isDefault ? "★ " : "  "}
            {m.name} — {m.id} ({m.status})
          </Text>
        ))}
      <Text color={theme.muted}>Import, export, and version models here (v3.3+).</Text>
    </Box>
  );
}
