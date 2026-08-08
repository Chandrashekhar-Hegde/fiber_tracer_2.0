export interface AnalysisConfig {
  dataPath: string;
  outputDir: string;
  voxelSpacing: [number, number, number];
  fiberDiameter: number;
  regime: "auto" | "resolved" | "marginal" | "subvoxel";
  method: "otsu" | "watershed" | "unet";
  thresholdMethod: "otsu" | "manual" | "adaptive" | "multiotsu";
  thresholdValue?: number;
  model: string;
  batchSize: number;
  computeMorphometry: boolean;
  computeOrientationTensor: boolean;
  computeTracking: boolean;
  computeTda: boolean;
  computeDvc: boolean;
  dvcReferencePath: string;
  dvcDeformedPath: string;
  computeDic: boolean;
  dicReferencePath: string;
  dicDeformedPath: string;
  computeTwin: boolean;
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

export type ModelSource = "bundled" | "local" | "remote";
export type ModelStatus =
  | "ready"
  | "planned"
  | "downloading"
  | "error"
  | "missing"
  | "loading";
export type ExperimentType = "train" | "analyze";
export type ExperimentStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";
export type TrainingDevice = "auto" | "cpu" | "cuda" | "mps";

export interface Model {
  id: string;
  name: string;
  architecture: string;
  source: ModelSource;
  path: string;
  version: string;
  createdAt: string;
  tags: string[];
  description: string;
  status: ModelStatus;
  isDefault: boolean;
}

export interface Experiment {
  id: string;
  name: string;
  type: ExperimentType;
  modelId: string;
  dataset: string;
  configSnapshot: Record<string, unknown>;
  status: ExperimentStatus;
  metrics: Record<string, unknown>;
  history: Record<string, unknown>;
  startedAt: string;
  finishedAt?: string;
  artifactDir: string;
  errorMessage?: string;
}

export interface TrainingOptions {
  datasetDir: string;
  modelId: string;
  outputDir: string;
  name: string;
  epochs: number;
  batchSize: number;
  lr: number;
  device: TrainingDevice;
}
