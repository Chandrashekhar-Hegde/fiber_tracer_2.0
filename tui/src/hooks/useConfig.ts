import { useState, useEffect } from "react";
import { loadConfig, saveConfig } from "../config";
import type { UserConfig } from "../types";

export function useConfig() {
  const [config, setConfig] = useState<UserConfig>(() => loadConfig());

  useEffect(() => {
    saveConfig(config);
  }, [config]);

  return { config, setConfig };
}
