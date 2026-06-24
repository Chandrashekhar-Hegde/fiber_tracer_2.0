import React, { forwardRef, useImperativeHandle, useState } from "react";
import { Box, Text } from "ink";
import { startTraining } from "../bridge";
import type { Theme } from "../theme";
import type { ProgressEvent, TrainingOptions } from "../types";

interface TrainingProps {
  theme: Theme;
}

export interface TrainingRef {
  start: () => void;
}

const TRAINING_OPTIONS: TrainingOptions = {
  datasetDir: "./data/patches",
  modelId: "unet-v3.2",
  outputDir: "./experiments/quick-test",
  name: "Quick test",
  epochs: 2,
  batchSize: 2,
  lr: 1e-3,
  device: "auto",
};

export const Training = forwardRef<TrainingRef, TrainingProps>(
  function Training({ theme }, ref) {
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState<ProgressEvent | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);

    const handleStart = () => {
      if (running) return;
      setRunning(true);
      setProgress(null);
      setLogs([]);
      setError(null);

      startTraining(TRAINING_OPTIONS, {
        onProgress: (event) => setProgress(event),
        onLog: (line) => setLogs((prev) => [...prev.slice(-49), line]),
      })
        .then((result) => {
          if (!result.success) {
            setError(result.error ?? "Training failed");
          }
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        })
        .finally(() => {
          setRunning(false);
        });
    };

    useImperativeHandle(ref, () => ({
      start: handleStart,
    }));

    return (
      <Box flexDirection="column">
        <Text bold color={theme.accent}>Training</Text>
        <Text color={theme.foreground}>Task: Semantic segmentation fine-tuning</Text>
        <Text color={theme.foreground}>Model: {TRAINING_OPTIONS.modelId}</Text>
        {running && <Text color={theme.muted}>Running...</Text>}
        {progress && (
          <Text color={theme.foreground}>
            Progress: {progress.percent}% — {progress.message}
          </Text>
        )}
        {error && <Text color={theme.error}>Error: {error}</Text>}
        {logs.length > 0 && (
          <Box flexDirection="column" marginTop={1}>
            <Text bold color={theme.accent}>Recent logs</Text>
            {logs.slice(-10).map((line, index) => (
              <Text key={index} color={theme.muted}>
                {line}
              </Text>
            ))}
          </Box>
        )}
        <Text color={theme.muted}>Press "s" to start training.</Text>
      </Box>
    );
  }
);
