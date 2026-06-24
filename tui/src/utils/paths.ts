import { mkdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";

let cachedConfigDir: string | undefined;

export function getConfigDir(): string {
  if (cachedConfigDir) return cachedConfigDir;

  const dir = process.env.FIBER_TRACER_CONFIG_DIR || join(homedir(), ".config", "fiber-tracer");
  mkdirSync(dir, { recursive: true });
  cachedConfigDir = dir;
  return dir;
}
