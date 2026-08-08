import { describe, expect, it, mock, afterEach } from "bun:test";
import { EventEmitter } from "events";
import * as childProcess from "child_process";
import { runAnalysis } from "./bridge";

const originalSpawn = childProcess.spawn;

function createMockProcess(options: {
  code?: number;
  stdout?: string;
  stderr?: string;
  chunks?: string[];
  error?: Error;
}) {
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  const proc = new EventEmitter() as any;
  proc.stdout = stdout;
  proc.stderr = stderr;

  setTimeout(() => {
    if (options.error) {
      proc.emit("error", options.error);
      return;
    }
    if (options.chunks && options.chunks.length > 0) {
      options.chunks.forEach((chunk, index) => {
        setTimeout(() => stdout.emit("data", Buffer.from(chunk)), index * 5);
      });
      setTimeout(() => proc.emit("close", options.code ?? 0), options.chunks.length * 5 + 5);
    } else {
      if (options.stdout) stdout.emit("data", Buffer.from(options.stdout));
      if (options.stderr) stderr.emit("data", Buffer.from(options.stderr));
      proc.emit("close", options.code ?? 0);
    }
  }, 0);

  return proc;
}

function setMockSpawn(proc: any) {
  mock.module("child_process", () => ({ ...childProcess, spawn: () => proc }));
}

afterEach(() => {
  mock.module("child_process", () => ({ ...childProcess, spawn: originalSpawn }));
});

describe("bridge", () => {
  it("reports error when fiber-tracer CLI is missing", async () => {
    const proc = createMockProcess({ error: new Error("spawn fiber-tracer ENOENT") });
    setMockSpawn(proc);

    const result = await runAnalysis({
      dataPath: "/nonexistent",
      outputDir: "/tmp/out",
      voxelSpacing: [1, 1, 1],
      fiberDiameter: 10,
      regime: "auto",
      method: "otsu",
      thresholdMethod: "otsu",
      model: "",
      batchSize: 1,
      computeMorphometry: true,
      computeOrientationTensor: true,
      computeTracking: true,
      computeTda: false,
      computeDvc: false,
      dvcReferencePath: "",
      dvcDeformedPath: "",
      computeDic: false,
      dicReferencePath: "",
      dicDeformedPath: "",
      computeTwin: false,
    });
    expect(result.success).toBe(false);
    expect(result.error).toContain("ENOENT");
  });
});

describe("listModels", () => {
  it("maps snake_case CLI output to camelCase Model objects", async () => {
    const raw = [
      {
        model_id: "unet-v3.2",
        name: "Fiber U-Net v3.2",
        architecture: "unet3d",
        source: "bundled",
        path: "models/fiber_unet_v2_full.pt",
        version: "3.2.0",
        created_at: "2024-01-01T00:00:00Z",
        tags: [],
        description: "Default model",
        status: "ready",
        is_default: true,
      },
    ];
    const proc = createMockProcess({ code: 0, stdout: JSON.stringify(raw) });
    setMockSpawn(proc);

    const { listModels } = await import("./bridge");
    const models = await listModels();

    expect(models).toHaveLength(1);
    expect(models[0].id).toBe("unet-v3.2");
    expect(models[0].isDefault).toBe(true);
    expect(models[0].createdAt).toBe("2024-01-01T00:00:00Z");
    expect(models[0].name).toBe("Fiber U-Net v3.2");
    expect("modelId" in models[0]).toBe(false);
  });
});

describe("listExperiments", () => {
  it("maps snake_case CLI output to camelCase Experiment objects", async () => {
    const raw = [
      {
        id: "exp-20240101-abc123",
        name: "train-unet-v3.2",
        type: "train",
        model_id: "unet-v3.2",
        dataset: "/data/train",
        config_snapshot: { epochs: 10, batch_size: 4 },
        status: "completed",
        metrics: {},
        history: {},
        started_at: "2024-01-01T10:00:00Z",
        finished_at: "2024-01-01T11:00:00Z",
        artifact_dir: "/output/exp-1",
        error_message: "",
      },
    ];
    const proc = createMockProcess({ code: 0, stdout: JSON.stringify(raw) });
    setMockSpawn(proc);

    const { listExperiments } = await import("./bridge");
    const experiments = await listExperiments();

    expect(experiments).toHaveLength(1);
    expect(experiments[0].id).toBe("exp-20240101-abc123");
    expect(experiments[0].modelId).toBe("unet-v3.2");
    expect(experiments[0].configSnapshot).toEqual({ epochs: 10, batchSize: 4 });
    expect(experiments[0].startedAt).toBe("2024-01-01T10:00:00Z");
    expect(experiments[0].finishedAt).toBe("2024-01-01T11:00:00Z");
    expect(experiments[0].artifactDir).toBe("/output/exp-1");
  });
});

describe("startTraining", () => {
  const trainingOptions = {
    datasetDir: "/data/train",
    modelId: "unet-v3.2",
    outputDir: "/output/exp-1",
    name: "exp-1",
    epochs: 10,
    batchSize: 4,
    lr: 1e-3,
    device: "auto" as const,
  };

  it("streams progress and resolves on success", async () => {
    const progress = [
      JSON.stringify({ stage: "train", percent: 0, elapsedSeconds: 0, message: "start" }),
      JSON.stringify({ stage: "train", percent: 50, elapsedSeconds: 60, message: "halfway" }),
      JSON.stringify({ stage: "train", percent: 100, elapsedSeconds: 120, message: "done" }),
    ];
    const proc = createMockProcess({ code: 0, chunks: progress.map((line) => `${line}\n`) });
    setMockSpawn(proc);

    const { startTraining } = await import("./bridge");
    const onProgress = mock((_: import("./types").ProgressEvent) => {});
    const onLog = mock((_: string) => {});

    const result = await startTraining(trainingOptions, { onProgress, onLog });

    expect(result.success).toBe(true);
    expect(result.outputDir).toBe("/output/exp-1");
    expect(onProgress).toHaveBeenCalledTimes(3);
    expect(onProgress.mock.calls[0][0].percent).toBe(0);
    expect(onProgress.mock.calls[2][0].percent).toBe(100);
    expect(onLog).toHaveBeenCalledTimes(3);
  });

  it("parses multiple JSON progress events in a single chunk", async () => {
    const progress = [
      JSON.stringify({ stage: "train", percent: 10, elapsedSeconds: 10, message: "a" }),
      JSON.stringify({ stage: "train", percent: 20, elapsedSeconds: 20, message: "b" }),
      JSON.stringify({ stage: "train", percent: 30, elapsedSeconds: 30, message: "c" }),
    ];
    const proc = createMockProcess({ code: 0, chunks: [progress.map((line) => `${line}`).join("\n") + "\n"] });
    setMockSpawn(proc);

    const { startTraining } = await import("./bridge");
    const onProgress = mock((_: import("./types").ProgressEvent) => {});

    const result = await startTraining(trainingOptions, { onProgress });

    expect(result.success).toBe(true);
    expect(onProgress).toHaveBeenCalledTimes(3);
    expect(onProgress.mock.calls[0][0].percent).toBe(10);
    expect(onProgress.mock.calls[1][0].percent).toBe(20);
    expect(onProgress.mock.calls[2][0].percent).toBe(30);
  });

  it("resolves with error when training fails", async () => {
    const proc = createMockProcess({ code: 1, stderr: "Out of memory" });
    setMockSpawn(proc);

    const { startTraining } = await import("./bridge");
    const result = await startTraining(trainingOptions);

    expect(result.success).toBe(false);
    expect(result.outputDir).toBe("/output/exp-1");
    expect(result.error).toContain("Out of memory");
  });
});
