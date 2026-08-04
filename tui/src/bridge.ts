import { spawn } from "child_process";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import type {
  AnalysisConfig,
  BridgeResult,
  Experiment,
  Model,
  ProgressEvent,
  TrainingOptions,
} from "./types";

export interface BridgeOptions {
  onProgress?: (event: ProgressEvent) => void;
  onLog?: (line: string) => void;
}

export async function runAnalysis(
  config: AnalysisConfig,
  options: BridgeOptions = {}
): Promise<BridgeResult> {
  const tmpDir = mkdtempSync(join(tmpdir(), "ft-"));
  const tmpConfig = join(tmpDir, "config.json");
  writeFileSync(tmpConfig, buildJson(config));

  const cleanup = () => {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  };

  try {
    const result = await runStreaming(
      ["run", "--config", tmpConfig],
      config.outputDir,
      options
    );
    cleanup();
    return result;
  } catch (err) {
    cleanup();
    return {
      success: false,
      outputDir: config.outputDir,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

function buildJson(config: AnalysisConfig): string {
  return JSON.stringify({
    data_path: config.dataPath,
    output_dir: config.outputDir,
    voxel_spacing_um: config.voxelSpacing,
    fiber_diameter_um: config.fiberDiameter,
    regime: config.regime,
    segmentation: {
      method: config.method,
      model_path: config.model,
      batch_size: config.batchSize,
      threshold_method: config.thresholdMethod,
      threshold_value: config.thresholdValue ?? null,
    },
    analysis: {
      compute_morphometry: config.computeMorphometry,
      compute_orientation_tensor: config.computeOrientationTensor,
      compute_tracking: config.computeTracking,
      compute_tda_descriptors: config.computeTda,
    },
    dvc: {
      enabled: config.computeDvc,
      reference_path: config.dvcReferencePath,
      deformed_path: config.dvcDeformedPath,
    },
    dic: {
      enabled: config.computeDic,
      reference_path: config.dicReferencePath,
      deformed_path: config.dicDeformedPath,
    },
  });
}

function parseProgressLine(line: string): ProgressEvent | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    const data = JSON.parse(trimmed) as Partial<ProgressEvent>;
    if (data.stage && typeof data.percent === "number") return data as ProgressEvent;
  } catch {}
  return null;
}

function runStreaming(
  args: string[],
  outputDir: string,
  options: BridgeOptions = {}
): Promise<BridgeResult> {
  return new Promise((resolve, reject) => {
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawn("fiber-tracer", args, {
        env: { ...process.env, PYTHONUNBUFFERED: "1", FIBER_TRACER_JSON_PROGRESS: "1" },
      });
    } catch (err) {
      reject(err instanceof Error ? err : new Error(String(err)));
      return;
    }

    let stdout = "";
    let stderr = "";
    let stdoutBuffer = "";
    let stderrBuffer = "";

    const handleLines = (buffer: string, isStdout: boolean): string => {
      const lines = buffer.split("\n");
      const remainder = lines.pop() ?? "";
      for (const line of lines) {
        if (line.length === 0) continue;
        if (isStdout) {
          stdout += line + "\n";
          const event = parseProgressLine(line);
          if (event && options.onProgress) options.onProgress(event);
        } else {
          stderr += line + "\n";
        }
        if (options.onLog) options.onLog(line);
      }
      return remainder;
    };

    proc.stdout!.on("data", (data) => {
      stdoutBuffer += data.toString();
      stdoutBuffer = handleLines(stdoutBuffer, true);
    });

    proc.stderr!.on("data", (data) => {
      stderrBuffer += data.toString();
      stderrBuffer = handleLines(stderrBuffer, false);
    });

    proc.on("error", (err) => {
      reject(err);
    });

    proc.on("close", (code) => {
      if (stdoutBuffer.length > 0) {
        handleLines(stdoutBuffer + "\n", true);
      }
      if (stderrBuffer.length > 0) {
        handleLines(stderrBuffer + "\n", false);
      }
      if (code === 0) {
        resolve({ success: true, outputDir });
      } else {
        resolve({ success: false, outputDir, error: stderr.trim() || stdout.trim() });
      }
    });
  });
}

export function toCamelCase(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map(toCamelCase);
  }
  if (obj !== null && typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      result[toCamelCaseKey(key)] = toCamelCase(value);
    }
    return result;
  }
  return obj;
}

function toCamelCaseKey(key: string): string {
  return key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

function spawnJsonList<T>(args: string[]): Promise<T[]> {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";

    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawn("fiber-tracer", args, { env: process.env });
    } catch (err) {
      reject(err instanceof Error ? err : new Error(String(err)));
      return;
    }

    proc.stdout!.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr!.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      reject(err);
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || stdout.trim() || `Command exited with code ${code}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim());
        if (!Array.isArray(parsed)) {
          reject(
            new Error(
              `Expected JSON array from CLI, got ${typeof parsed}. stdout: ${stdout.trim()} stderr: ${stderr.trim()}`
            )
          );
          return;
        }
        resolve(parsed.map(toCamelCase) as T[]);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse JSON output: ${err instanceof Error ? err.message : String(err)}. stdout: ${stdout.trim()} stderr: ${stderr.trim()}`
          )
        );
      }
    });
  });
}

export async function listModels(): Promise<Model[]> {
  const models = await spawnJsonList<Model>(["model", "list", "--json"]);
  return models.map((model) => {
    const m = model as unknown as Record<string, unknown>;
    const { modelId, ...rest } = m;
    return { ...rest, id: modelId } as unknown as Model;
  });
}

export async function listExperiments(): Promise<Experiment[]> {
  return spawnJsonList<Experiment>(["experiment", "list", "--json"]);
}

export async function startTraining(
  options: TrainingOptions,
  callbacks: BridgeOptions = {}
): Promise<BridgeResult> {
  const args = [
    "train",
    "--dataset-dir",
    options.datasetDir,
    "--model-id",
    options.modelId,
    "--output-dir",
    options.outputDir,
    "--epochs",
    String(options.epochs),
    "--batch-size",
    String(options.batchSize),
    "--lr",
    String(options.lr),
    "--device",
    options.device,
  ];
  if (options.name) {
    args.push("--name", options.name);
  }

  return runStreaming(args, options.outputDir, callbacks);
}
