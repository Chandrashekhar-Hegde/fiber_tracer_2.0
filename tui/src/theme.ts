import { readFileSync, existsSync } from "fs";
import { join } from "path";
import { getConfigDir } from "./utils/paths";

export interface Theme {
  name: string;
  background: string;
  foreground: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  muted: string;
  border: string;
  highlight: string;
}

const THEME_KEYS: (keyof Theme)[] = [
  "name",
  "background",
  "foreground",
  "accent",
  "success",
  "warning",
  "error",
  "muted",
  "border",
  "highlight",
];

export const DEFAULT_THEME: Theme = {
  name: "Dracula",
  background: "#282a36",
  foreground: "#f8f8f2",
  accent: "#bd93f9",
  success: "#50fa7b",
  warning: "#ffb86c",
  error: "#ff5555",
  muted: "#6272a4",
  border: "#44475a",
  highlight: "#ff79c6",
};

export function isTheme(obj: unknown): obj is Theme {
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
    return false;
  }
  const record = obj as Record<string, unknown>;
  return THEME_KEYS.every((key) => typeof record[key] === "string");
}

function parseTheme(data: string): Theme | null {
  try {
    const parsed = JSON.parse(data) as unknown;
    if (isTheme(parsed)) return parsed;
  } catch {
    // ignore malformed JSON
  }
  return null;
}

function loadBuiltinTheme(name: string): Theme {
  const path = join(import.meta.dirname, "../themes", `${name}.json`);
  if (!existsSync(path)) return DEFAULT_THEME;
  return parseTheme(readFileSync(path, "utf8")) ?? DEFAULT_THEME;
}

export const BUILTIN_THEMES: Record<string, Theme> = {
  dracula: loadBuiltinTheme("dracula"),
  "catppuccin-mocha": loadBuiltinTheme("catppuccin-mocha"),
  "catppuccin-latte": loadBuiltinTheme("catppuccin-latte"),
  "one-dark": loadBuiltinTheme("one-dark"),
  nord: loadBuiltinTheme("nord"),
};

export function loadTheme(name: string): Theme {
  if (BUILTIN_THEMES[name]) return BUILTIN_THEMES[name];

  const custom = join(getConfigDir(), "themes", `${name}.json`);
  if (existsSync(custom)) {
    const theme = parseTheme(readFileSync(custom, "utf8"));
    if (theme) return theme;
  }

  return DEFAULT_THEME;
}
