import { spawn } from "child_process";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import type { AnalysisConfig, BridgeResult, ProgressEvent } from "./types";

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
