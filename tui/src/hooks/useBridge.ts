import { useState, useCallback } from "react";
import { runAnalysis } from "../bridge";
import type { AnalysisConfig, BridgeStatus, ProgressEvent } from "../types";

export function useBridge() {
  const [status, setStatus] = useState<BridgeStatus>("idle");
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (config: AnalysisConfig) => {
    setStatus("running");
    setProgress(null);
    setLogs([]);
    setError(null);

    const result = await runAnalysis(config, {
      onProgress: setProgress,
      onLog: (line) => setLogs((prev) => [...prev, line]),
    });

    setStatus(result.success ? "success" : "error");
    if (result.error) setError(result.error);
    return result;
  }, []);

  return { status, progress, logs, error, run };
}
