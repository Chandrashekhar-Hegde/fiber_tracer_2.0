export interface AnalysisConfig {
  dataPath: string;
  outputDir: string;
  voxelSpacing: [number, number, number];
  fiberDiameter: number;
  regime: "auto" | "resolved" | "marginal" | "subvoxel";
  method: "otsu" | "watershed" | "unet";
  model: string;
  batchSize: number;
  computeMorphometry: boolean;
  computeOrientationTensor: boolean;
  computeTda: boolean;
}

export interface RunRecord {
  id: string;
  name: string;
  config: AnalysisConfig;
  status: "running" | "success" | "failed" | "cancelled";
  startedAt: string;
  finishedAt?: string;
  outputDir: string;
  summary?: Record<string, unknown>;
}

export interface UserConfig {
  theme: string;
  defaultOutputDir: string;
  defaultModel: string;
  logLevel: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  vimBindings: boolean;
}

export interface ProgressEvent {
  stage: string;
  percent: number;
  elapsedSeconds: number;
  message: string;
}

export interface BridgeResult {
  success: boolean;
  outputDir: string;
  summary?: Record<string, unknown>;
  error?: string;
}

export type BridgeStatus = "idle" | "running" | "success" | "error" | "cancelled";
