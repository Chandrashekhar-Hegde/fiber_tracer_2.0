import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import { getConfigDir } from "./utils/paths";
import type { UserConfig } from "./types";

const CONFIG_PATH = join(getConfigDir(), "tui-config.json");

export const DEFAULT_CONFIG: UserConfig = {
  theme: "dracula",
  defaultOutputDir: "./results",
  defaultModel: "models/fiber_unet_v2_full.pt",
  logLevel: "INFO",
  vimBindings: true,
};

export function loadConfig(): UserConfig {
  if (!existsSync(CONFIG_PATH)) return DEFAULT_CONFIG;
  try {
    const data = JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as Partial<UserConfig>;
    return { ...DEFAULT_CONFIG, ...data };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function saveConfig(config: UserConfig): void {
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}
