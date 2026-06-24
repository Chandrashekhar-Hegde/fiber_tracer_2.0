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
  const tmpYaml = join(tmpDir, "config.yaml");
  writeFileSync(tmpYaml, buildYaml(config));

  const cleanup = () => {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  };

  return new Promise((resolve) => {
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawn("fiber-tracer", ["run", "--config", tmpYaml], {
        env: { ...process.env, PYTHONUNBUFFERED: "1", FIBER_TRACER_JSON_PROGRESS: "1" },
      });
    } catch (err) {
      cleanup();
      resolve({
        success: false,
        outputDir: config.outputDir,
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }

    let stdout = "";
    let stderr = "";

    proc.stdout!.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      if (options.onLog) options.onLog(text.trim());
      const event = parseProgress(text);
      if (event && options.onProgress) options.onProgress(event);
    });

    proc.stderr!.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      if (options.onLog) options.onLog(text.trim());
    });

    proc.on("error", (err) => {
      cleanup();
      resolve({
        success: false,
        outputDir: config.outputDir,
        error: err instanceof Error ? err.message : String(err),
      });
    });

    proc.on("close", (code) => {
      cleanup();
      if (code === 0) {
        resolve({ success: true, outputDir: config.outputDir });
      } else {
        resolve({ success: false, outputDir: config.outputDir, error: stderr || stdout });
      }
    });
  });
}

function buildYaml(config: AnalysisConfig): string {
  return `
data_path: ${JSON.stringify(config.dataPath)}
output_dir: ${JSON.stringify(config.outputDir)}
voxel_spacing_um: [${config.voxelSpacing.join(", ")}]
fiber_diameter_um: ${config.fiberDiameter}
regime: ${config.regime}
segmentation:
  method: ${config.method}
  model_path: ${JSON.stringify(config.model)}
  batch_size: ${config.batchSize}
analysis:
  compute_morphometry: ${config.computeMorphometry}
  compute_orientation_tensor: ${config.computeOrientationTensor}
  compute_tda_descriptors: ${config.computeTda}
`.trim();
}

function parseProgress(text: string): ProgressEvent | null {
  const match = text.match(/\{[^}]+\}/);
  if (!match) return null;
  try {
    const data = JSON.parse(match[0]) as Partial<ProgressEvent>;
    if (data.stage && typeof data.percent === "number") return data as ProgressEvent;
  } catch {}
  return null;
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
        const items = Array.isArray(parsed) ? parsed : [];
        resolve(items.map(toCamelCase) as T[]);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse JSON output: ${err instanceof Error ? err.message : String(err)}`
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
    if ("modelId" in m) {
      m.id = m.modelId;
      delete m.modelId;
    }
    return m as unknown as Model;
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

  return new Promise((resolve) => {
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawn("fiber-tracer", args, {
        env: { ...process.env, PYTHONUNBUFFERED: "1", FIBER_TRACER_JSON_PROGRESS: "1" },
      });
    } catch (err) {
      resolve({
        success: false,
        outputDir: options.outputDir,
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }

    let stdout = "";
    let stderr = "";

    proc.stdout!.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      if (callbacks.onLog) callbacks.onLog(text.trim());
      const event = parseProgress(text);
      if (event && callbacks.onProgress) callbacks.onProgress(event);
    });

    proc.stderr!.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      if (callbacks.onLog) callbacks.onLog(text.trim());
    });

    proc.on("error", (err) => {
      resolve({
        success: false,
        outputDir: options.outputDir,
        error: err instanceof Error ? err.message : String(err),
      });
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve({ success: true, outputDir: options.outputDir });
      } else {
        resolve({ success: false, outputDir: options.outputDir, error: stderr || stdout });
      }
    });
  });
}
