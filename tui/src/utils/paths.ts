import { mkdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";

export function getConfigDir(): string {
  const dir = process.env.FIBER_TRACER_CONFIG_DIR || join(homedir(), ".config", "fiber-tracer");
  mkdirSync(dir, { recursive: true });
  return dir;
}
