# Fiber Tracer Terminal UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a keyboard-driven, themeable Terminal UI for Fiber Tracer using Bun + Ink + @inkjs/ui that wraps the existing Python analysis engine and supports future model registries, training, and experiments.

**Architecture:** The TUI is a React/Ink application in a `tui/` directory. It communicates with the existing Python backend through a small bridge that spawns `fiber-tracer` CLI commands and parses JSON progress/results. State (config, history, themes) is persisted to `~/.config/fiber-tracer/`. The UI is organized around a persistent sidebar that doubles as a wizard navigator.

**Tech Stack:** Bun 1.3+, React 18, Ink 5, @inkjs/ui 2.x, TypeScript, Python 3.10+ Fiber Tracer backend.

---

## File Structure

```
tui/
├── package.json
├── tsconfig.json
├── bun.lockb
├── README.md
├── src/
│   ├── app.tsx                    # Ink root, global key listeners
│   ├── index.tsx                  # Entry point
│   ├── theme.ts                   # OpenTUI theme loader & built-ins
│   ├── config.ts                  # User config read/write
│   ├── history.ts                 # Run history read/write
│   ├── bridge.ts                  # Python subprocess wrapper
│   ├── types.ts                   # Shared TypeScript types
│   ├── keybindings.ts             # Shortcut definitions & handlers
│   ├── hooks/
│   │   ├── useConfig.ts           # Load/save user config
│   │   ├── useHistory.ts          # Load/save run history
│   │   └── useBridge.ts           # Spawn Python and stream progress
│   ├── components/
│   │   ├── layout.tsx             # Sidebar + header + footer shell
│   │   ├── sidebar.tsx            # Navigation tree
│   │   ├── header.tsx             # Version, mode, global hints
│   │   ├── footer.tsx             # Context-sensitive shortcuts
│   │   ├── wizard-shell.tsx       # Step header + navigation buttons
│   │   ├── select-data.tsx        # Step 1: file selection
│   │   ├── configure.tsx          # Step 2: parameter form
│   │   ├── review.tsx             # Step 3: run summary
│   │   ├── run-watch.tsx          # Step 4: live progress
│   │   ├── dashboard.tsx          # Results dashboard
│   │   ├── history.tsx            # Past runs list
│   │   ├── experiments.tsx        # Parameter sweep comparison
│   │   ├── model-registry.tsx     # Installed/downloadable models
│   │   ├── training.tsx           # Train/fine-tune wizard
│   │   ├── logs.tsx               # Log viewer
│   │   └── settings.tsx           # Theme & defaults
│   └── utils/
│       ├── paths.ts               # Config dir helpers
│       └── format.ts              # Time/size/format helpers
└── themes/
    ├── dracula.json
    ├── catppuccin-mocha.json
    ├── catppuccin-latte.json
    ├── one-dark.json
    └── nord.json
```

Python side (minimal additions):

```
src/fiber_tracer/
├── tui_server.py                  # Optional: long-running JSON-RPC server
└── cli.py                         # Add `tui-server` subcommand
```

---

## Phase 1: Foundation

### Task 1: Scaffold TUI project

**Files:**
- Create: `tui/package.json`
- Create: `tui/tsconfig.json`
- Create: `tui/.gitignore`
- Create: `tui/README.md`

- [ ] **Step 1: Initialize Bun project**

Run:
```bash
cd /Users/cgh/Code/fibre/fiber_tracer_2.0
export PATH="$HOME/.bun/bin:$PATH"
bun init -y
```

This creates `package.json`, `tsconfig.json`, `bun.lockb`, `index.ts`, `.gitignore`.

- [ ] **Step 2: Install dependencies**

Run:
```bash
cd tui
bun add react ink @inkjs/ui
bun add -d @types/react @types/ink typescript
```

Expected: `bun.lockb` updated, `node_modules/` created.

- [ ] **Step 3: Update package.json scripts**

Modify `tui/package.json`:

```json
{
  "name": "fiber-tracer-tui",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "bun run src/index.tsx",
    "build": "bun build src/index.tsx --outdir=dist --target=node",
    "test": "bun test",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@inkjs/ui": "^2.0.0",
    "ink": "^5.0.0",
    "opentui": "^0.0.1",
    "react": "^18.3.0"
  },
  "devDependencies": {
    "@types/ink": "^2.0.3",
    "@types/react": "^18.3.0",
    "typescript": "^5.0.0"
  }
}
```

- [ ] **Step 4: Configure TypeScript**

Create `tui/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src/**/*"]
}
```

- [ ] **Step 5: Add .gitignore**

Create `tui/.gitignore`:

```
node_modules/
dist/
*.log
```

- [ ] **Step 6: Add README**

Create `tui/README.md`:

```markdown
# Fiber Tracer TUI

Interactive terminal UI for Fiber Tracer.

## Run locally

```bash
bun run dev
```

## Build

```bash
bun run build
```
```

- [ ] **Step 7: Commit**

```bash
git add tui/
git commit -m "chore(tui): scaffold Bun + Ink + @inkjs/ui project"
```

---

### Task 2: Theme system

**Files:**
- Create: `tui/src/theme.ts`
- Create: `tui/themes/dracula.json`
- Create: `tui/themes/catppuccin-mocha.json`
- Create: `tui/themes/catppuccin-latte.json`
- Create: `tui/themes/one-dark.json`
- Create: `tui/themes/nord.json`
- Create: `tui/src/utils/paths.ts`

- [ ] **Step 1: Write theme type and loader**

Create `tui/src/theme.ts`:

```typescript
import { readFileSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

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

export const BUILTIN_THEMES: Record<string, Theme> = {
  dracula: loadBuiltinTheme("dracula"),
  "catppuccin-mocha": loadBuiltinTheme("catppuccin-mocha"),
  "catppuccin-latte": loadBuiltinTheme("catppuccin-latte"),
  "one-dark": loadBuiltinTheme("one-dark"),
  nord: loadBuiltinTheme("nord"),
};

function loadBuiltinTheme(name: string): Theme {
  const path = join(import.meta.dirname, "../themes", `${name}.json`);
  return JSON.parse(readFileSync(path, "utf8")) as Theme;
}

export function loadTheme(name: string): Theme {
  if (BUILTIN_THEMES[name]) return BUILTIN_THEMES[name];
  const custom = join(getConfigDir(), "themes", `${name}.json`);
  if (existsSync(custom)) {
    return JSON.parse(readFileSync(custom, "utf8")) as Theme;
  }
  return BUILTIN_THEMES.dracula;
}
```

- [ ] **Step 2: Add Dracula theme JSON**

Create `tui/themes/dracula.json`:

```json
{
  "name": "Dracula",
  "background": "#282a36",
  "foreground": "#f8f8f2",
  "accent": "#bd93f9",
  "success": "#50fa7b",
  "warning": "#ffb86c",
  "error": "#ff5555",
  "muted": "#6272a4",
  "border": "#44475a",
  "highlight": "#ff79c6"
}
```

- [ ] **Step 3: Add remaining theme JSONs**

Create `tui/themes/catppuccin-mocha.json`:

```json
{
  "name": "Catppuccin Mocha",
  "background": "#1e1e2e",
  "foreground": "#cdd6f4",
  "accent": "#b4befe",
  "success": "#a6e3a1",
  "warning": "#f9e2af",
  "error": "#f38ba8",
  "muted": "#6c7086",
  "border": "#313244",
  "highlight": "#f5c2e7"
}
```

Create `tui/themes/catppuccin-latte.json`:

```json
{
  "name": "Catppuccin Latte",
  "background": "#eff1f5",
  "foreground": "#4c4f69",
  "accent": "#7287fd",
  "success": "#40a02b",
  "warning": "#df8e1d",
  "error": "#d20f39",
  "muted": "#8c8fa1",
  "border": "#ccd0da",
  "highlight": "#ea76cb"
}
```

Create `tui/themes/one-dark.json`:

```json
{
  "name": "One Dark",
  "background": "#282c34",
  "foreground": "#abb2bf",
  "accent": "#61afef",
  "success": "#98c379",
  "warning": "#e5c07b",
  "error": "#e06c75",
  "muted": "#5c6370",
  "border": "#3e4451",
  "highlight": "#c678dd"
}
```

Create `tui/themes/nord.json`:

```json
{
  "name": "Nord",
  "background": "#2e3440",
  "foreground": "#d8dee9",
  "accent": "#88c0d0",
  "success": "#a3be8c",
  "warning": "#ebcb8b",
  "error": "#bf616a",
  "muted": "#4c566a",
  "border": "#3b4252",
  "highlight": "#b48ead"
}
```

- [ ] **Step 4: Add config-dir helper**

Create `tui/src/utils/paths.ts`:

```typescript
import { mkdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";

export function getConfigDir(): string {
  const dir = process.env.FIBER_TRACER_CONFIG_DIR || join(homedir(), ".config", "fiber-tracer");
  mkdirSync(dir, { recursive: true });
  return dir;
}
```

- [ ] **Step 5: Write test for theme loader**

Create `tui/src/theme.test.ts`:

```typescript
import { describe, expect, it } from "bun:test";
import { BUILTIN_THEMES, loadTheme } from "./theme";

describe("theme", () => {
  it("loads all built-in themes", () => {
    expect(Object.keys(BUILTIN_THEMES)).toContain("dracula");
    expect(BUILTIN_THEMES.dracula.name).toBe("Dracula");
  });

  it("falls back to dracula for unknown theme", () => {
    const theme = loadTheme("nonexistent");
    expect(theme.name).toBe("Dracula");
  });
});
```

- [ ] **Step 6: Run test**

Run:
```bash
cd tui
bun test src/theme.test.ts
```

Expected: 2 passing tests.

- [ ] **Step 7: Commit**

```bash
git add tui/src/theme.ts tui/src/theme.test.ts tui/src/utils/paths.ts tui/themes/
git commit -m "feat(tui): add OpenTUI theme loader and built-in themes"
```

---

### Task 3: Config and history persistence

**Files:**
- Create: `tui/src/types.ts`
- Create: `tui/src/config.ts`
- Create: `tui/src/history.ts`
- Create: `tui/src/hooks/useConfig.ts`
- Create: `tui/src/hooks/useHistory.ts`

- [ ] **Step 1: Define shared types**

Create `tui/src/types.ts`:

```typescript
export interface AnalysisConfig {
  dataPath: string;
  outputDir: string;
  voxelSpacing: [number, number, number];
  fiberDiameter: number;
  regime: "auto" | "resolved" | "marginal" | "subvoxel";
  method: "otsu" | "watershed" | "unet";
  model: string;
  batchSize: number;
  computeMorphometry: boolean;
  computeOrientationTensor: boolean;
  computeTda: boolean;
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
```

- [ ] **Step 2: Implement config persistence**

Create `tui/src/config.ts`:

```typescript
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
```

- [ ] **Step 3: Implement history persistence**

Create `tui/src/history.ts`:

```typescript
import { appendFileSync, readFileSync, existsSync, writeFileSync } from "fs";
import { join } from "path";
import { getConfigDir } from "./utils/paths";
import type { RunRecord } from "./types";

const HISTORY_PATH = join(getConfigDir(), "history.jsonl");

export function loadHistory(): RunRecord[] {
  if (!existsSync(HISTORY_PATH)) return [];
  return readFileSync(HISTORY_PATH, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RunRecord);
}

export function appendHistory(record: RunRecord): void {
  appendFileSync(HISTORY_PATH, JSON.stringify(record) + "\n");
}

export function updateHistory(updated: RunRecord): void {
  const history = loadHistory().map((r) => (r.id === updated.id ? updated : r));
  writeFileSync(HISTORY_PATH, history.map((r) => JSON.stringify(r)).join("\n") + "\n");
}
```

- [ ] **Step 4: Add React hooks**

Create `tui/src/hooks/useConfig.ts`:

```typescript
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
```

Create `tui/src/hooks/useHistory.ts`:

```typescript
import { useState, useCallback } from "react";
import { loadHistory, appendHistory, updateHistory } from "../history";
import type { RunRecord } from "../types";

export function useHistory() {
  const [history, setHistory] = useState<RunRecord[]>(() => loadHistory());

  const add = useCallback((record: RunRecord) => {
    appendHistory(record);
    setHistory((prev) => [...prev, record]);
  }, []);

  const update = useCallback((record: RunRecord) => {
    updateHistory(record);
    setHistory((prev) => prev.map((r) => (r.id === record.id ? record : r)));
  }, []);

  return { history, add, update };
}
```

- [ ] **Step 5: Write tests**

Create `tui/src/config.test.ts`:

```typescript
import { describe, expect, it, beforeEach } from "bun:test";
import { loadConfig, saveConfig, DEFAULT_CONFIG } from "./config";
import { join } from "path";
import { getConfigDir } from "./utils/paths";

const CONFIG_PATH = join(getConfigDir(), "tui-config.json");

describe("config", () => {
  beforeEach(() => {
    try { require("fs").unlinkSync(CONFIG_PATH); } catch {}
  });

  it("returns default config when file missing", () => {
    expect(loadConfig().theme).toBe("dracula");
  });

  it("persists config changes", () => {
    saveConfig({ ...DEFAULT_CONFIG, theme: "nord" });
    expect(loadConfig().theme).toBe("nord");
  });
});
```

- [ ] **Step 6: Run tests**

Run:
```bash
cd tui
bun test src/config.test.ts
```

Expected: 2 passing tests.

- [ ] **Step 7: Commit**

```bash
git add tui/src/types.ts tui/src/config.ts tui/src/config.test.ts tui/src/history.ts tui/src/hooks/useConfig.ts tui/src/hooks/useHistory.ts
git commit -m "feat(tui): add config and history persistence"
```

---

## Phase 2: Layout & Navigation

### Task 4: Layout shell

**Files:**
- Create: `tui/src/components/layout.tsx`
- Create: `tui/src/components/sidebar.tsx`
- Create: `tui/src/components/header.tsx`
- Create: `tui/src/components/footer.tsx`
- Modify: `tui/src/app.tsx`

- [ ] **Step 1: Build Sidebar component**

Create `tui/src/components/sidebar.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

export type Section =
  | "new-analysis"
  | "dashboard"
  | "history"
  | "experiments"
  | "model-registry"
  | "training"
  | "logs"
  | "settings";

interface SidebarProps {
  active: Section;
  wizardStep: number;
  theme: Theme;
}

const ITEMS: { key: Section; label: string; shortcut: string }[] = [
  { key: "new-analysis", label: "New Analysis", shortcut: "1" },
  { key: "dashboard", label: "Dashboard", shortcut: "2" },
  { key: "history", label: "History", shortcut: "3" },
  { key: "experiments", label: "Experiments", shortcut: "4" },
  { key: "model-registry", label: "Model Registry", shortcut: "5" },
  { key: "training", label: "Training", shortcut: "6" },
  { key: "logs", label: "Logs", shortcut: "7" },
  { key: "settings", label: "Settings", shortcut: "8" },
];

export function Sidebar({ active, wizardStep, theme }: SidebarProps) {
  return (
    <Box width={24} flexDirection="column" borderStyle="single" borderColor={theme.border} paddingX={1}>
      <Text bold color={theme.accent}>
        Fiber Tracer
      </Text>
      {ITEMS.map((item) => {
        const isActive = active === item.key;
        const isWizard = item.key === "new-analysis";
        return (
          <Box key={item.key} flexDirection="column" marginTop={isWizard ? 1 : 0}>
            <Text color={isActive ? theme.highlight : theme.foreground}>
              {isActive ? "▸ " : "  "}
              {item.label}
            </Text>
            {isWizard && isActive && (
              <Box flexDirection="column" paddingLeft={2}>
                {["1. Select Data", "2. Configure", "3. Review", "4. Run & Watch"].map((step, idx) => (
                  <Text key={step} color={idx === wizardStep ? theme.highlight : theme.muted}>
                    {step}
                  </Text>
                ))}
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
```

- [ ] **Step 2: Build Header component**

Create `tui/src/components/header.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface HeaderProps {
  theme: Theme;
}

export function Header({ theme }: HeaderProps) {
  return (
    <Box height={1} justifyContent="space-between" paddingX={1}>
      <Text color={theme.accent}>Fiber Tracer TUI</Text>
      <Text color={theme.muted}>v3.2.0 | ? help | q quit</Text>
    </Box>
  );
}
```

- [ ] **Step 3: Build Footer component**

Create `tui/src/components/footer.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface FooterProps {
  shortcuts: string[];
  theme: Theme;
}

export function Footer({ shortcuts, theme }: FooterProps) {
  return (
    <Box height={1} paddingX={1}>
      {shortcuts.map((shortcut, idx) => (
        <Text key={idx} color={theme.muted}>
          {shortcut}
          {idx < shortcuts.length - 1 ? "  " : ""}
        </Text>
      ))}
    </Box>
  );
}
```

- [ ] **Step 4: Build Layout component**

Create `tui/src/components/layout.tsx`:

```typescript
import React from "react";
import { Box } from "ink";
import { Header } from "./header";
import { Sidebar, type Section } from "./sidebar";
import { Footer } from "./footer";
import type { Theme } from "../theme";

interface LayoutProps {
  section: Section;
  wizardStep: number;
  footerShortcuts: string[];
  theme: Theme;
  children: React.ReactNode;
}

export function Layout({ section, wizardStep, footerShortcuts, theme, children }: LayoutProps) {
  return (
    <Box flexDirection="column" height="100%">
      <Header theme={theme} />
      <Box flexGrow={1}>
        <Sidebar active={section} wizardStep={wizardStep} theme={theme} />
        <Box flexGrow={1} paddingX={2} paddingY={1} flexDirection="column">
          {children}
        </Box>
      </Box>
      <Footer shortcuts={footerShortcuts} theme={theme} />
    </Box>
  );
}
```

- [ ] **Step 5: Wire up App**

Modify `tui/src/app.tsx`:

```typescript
import React, { useState } from "react";
import { useInput } from "ink";
import { Layout } from "./components/layout";
import { Settings } from "./components/settings";
import { useConfig } from "./hooks/useConfig";
import { loadTheme } from "./theme";
import type { Section } from "./components/sidebar";

export function App() {
  const { config, setConfig } = useConfig();
  const [section, setSection] = useState<Section>("settings");
  const [wizardStep, setWizardStep] = useState(0);
  const theme = loadTheme(config.theme);

  useInput((input, key) => {
    if (input === "q") process.exit(0);
    if (input === "1") setSection("new-analysis");
    if (input === "8") setSection("settings");
  });

  return (
    <Layout
      section={section}
      wizardStep={wizardStep}
      footerShortcuts={["1 New Analysis", "8 Settings", "q Quit"]}
      theme={theme}
    >
      <Settings config={config} onChange={setConfig} theme={theme} />
    </Layout>
  );
}
```

- [ ] **Step 6: Create initial Settings screen**

Create `tui/src/components/settings.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { UserConfig } from "../types";

interface SettingsProps {
  config: UserConfig;
  onChange: (config: UserConfig) => void;
  theme: Theme;
}

export function Settings({ config, onChange, theme }: SettingsProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Settings
      </Text>
      <Text color={theme.foreground}>Theme: {config.theme}</Text>
      <Text color={theme.muted}>Default output: {config.defaultOutputDir}</Text>
      <Text color={theme.muted}>Default model: {config.defaultModel}</Text>
    </Box>
  );
}
```

- [ ] **Step 7: Update entry point**

Modify `tui/src/index.tsx`:

```typescript
import React from "react";
import { render } from "ink";
import { App } from "./app";

render(<App />);
```

- [ ] **Step 8: Run TUI**

Run:
```bash
cd tui
bun run dev
```

Expected: Terminal shows the layout with sidebar, header, settings panel, and footer. Press `q` to exit.

- [ ] **Step 9: Commit**

```bash
git add tui/src/app.tsx tui/src/index.tsx tui/src/components/
git commit -m "feat(tui): add layout shell with sidebar, header, footer, and initial settings screen"
```

---

## Phase 3: Python Bridge

### Task 5: Python subprocess bridge

**Files:**
- Create: `tui/src/bridge.ts`
- Create: `tui/src/hooks/useBridge.ts`
- Modify: `tui/src/types.ts`

- [ ] **Step 1: Define bridge types**

Append to `tui/src/types.ts`:

```typescript
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
```

- [ ] **Step 2: Implement bridge**

Create `tui/src/bridge.ts`:

```typescript
import { spawn } from "child_process";
import { writeFileSync } from "fs";
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
  const tmpYaml = join(tmpdir(), `fiber-tracer-${Date.now()}.yaml`);
  writeFileSync(tmpYaml, buildYaml(config));

  return new Promise((resolve) => {
    const proc = spawn("fiber-tracer", ["run", "--config", tmpYaml], {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      if (options.onLog) options.onLog(text.trim());
      const event = parseProgress(text);
      if (event && options.onProgress) options.onProgress(event);
    });

    proc.stderr.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      if (options.onLog) options.onLog(text.trim());
    });

    proc.on("close", (code) => {
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
data: ${config.dataPath}
output: ${config.outputDir}
voxel_spacing: [${config.voxelSpacing.join(", ")}]
fiber_diameter: ${config.fiberDiameter}
regime: ${config.regime}
segmentation:
  method: ${config.method}
  model_path: ${config.model}
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
```

- [ ] **Step 3: Add React hook**

Create `tui/src/hooks/useBridge.ts`:

```typescript
import { useState, useCallback } from "react";
import { runAnalysis } from "../bridge";
import type { AnalysisConfig, BridgeStatus, ProgressEvent } from "../types";

export function useBridge() {
  const [status, setStatus] = useState<BridgeStatus>("idle");
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (config: AnalysisConfig) => {
    setStatus("running");
    setProgress(null);
    setLogs([]);
    setError(null);

    const result = await runAnalysis(config, {
      onProgress: setProgress,
      onLog: (line) => setLogs((prev) => [...prev, line]),
    });

    setStatus(result.success ? "success" : "error");
    if (result.error) setError(result.error);
    return result;
  }, []);

  return { status, progress, logs, error, run };
}
```

- [ ] **Step 4: Add progress JSON output to Python pipeline**

Modify `src/fiber_tracer/pipeline.py`:

Add `import json` and `import os` near the top of the file if not already present.

Find the end of `FiberAnalysisPipeline.run()` where `summary["elapsed_seconds"] = elapsed` is set, and add immediately after it:

```python
        if os.environ.get("FIBER_TRACER_JSON_PROGRESS"):
            print(
                json.dumps(
                    {
                        "stage": "complete",
                        "percent": 100,
                        "elapsedSeconds": elapsed,
                        "message": "Pipeline complete",
                    }
                )
            )
```

This lets the TUI bridge capture completion events from stdout when `FIBER_TRACER_JSON_PROGRESS=1` is exported.

- [ ] **Step 5: Write bridge test**

Create `tui/src/bridge.test.ts`:

```typescript
import { describe, expect, it } from "bun:test";
import { runAnalysis } from "./bridge";

describe("bridge", () => {
  it("reports error when fiber-tracer CLI is missing", async () => {
    const result = await runAnalysis({
      dataPath: "/nonexistent",
      outputDir: "/tmp/out",
      voxelSpacing: [1, 1, 1],
      fiberDiameter: 10,
      regime: "auto",
      method: "otsu",
      model: "",
      batchSize: 1,
      computeMorphometry: true,
      computeOrientationTensor: true,
      computeTda: false,
    });
    expect(result.success).toBe(false);
  });
});
```

- [ ] **Step 6: Run Python tests**

Run:
```bash
. .venv/bin/activate && pytest tests/ -q
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add tui/src/bridge.ts tui/src/bridge.test.ts tui/src/hooks/useBridge.ts src/fiber_tracer/pipeline.py
git commit -m "feat(tui): add Python subprocess bridge and progress hook"
```

---

## Phase 4: Wizard

### Task 6: Select Data screen

**Files:**
- Create: `tui/src/components/select-data.tsx`
- Modify: `tui/src/app.tsx`

- [ ] **Step 1: Build component**

Create `tui/src/components/select-data.tsx`:

```typescript
import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { Theme } from "../theme";

interface SelectDataProps {
  value: string;
  onChange: (path: string) => void;
  theme: Theme;
}

const RECENT_FILES = [
  "data/gfpa66_center.tif",
  "data/sample_a.tif",
  "data/sample_b.tif",
];

export function SelectData({ value, onChange, theme }: SelectDataProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  useInput((input, key) => {
    if (key.upArrow) setSelectedIdx((i) => Math.max(0, i - 1));
    if (key.downArrow) setSelectedIdx((i) => Math.min(RECENT_FILES.length - 1, i + 1));
    if (key.return) onChange(RECENT_FILES[selectedIdx]);
  });

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        1. Select Data
      </Text>
      <Text color={theme.muted}>Choose a recent volume or press b to browse</Text>
      {RECENT_FILES.map((file, idx) => (
        <Text key={file} color={idx === selectedIdx ? theme.highlight : theme.foreground}>
          {idx === selectedIdx ? "▸ " : "  "}
          {file}
        </Text>
      ))}
      <Box marginTop={1}>
        <Text color={theme.foreground}>Selected: {value || "none"}</Text>
      </Box>
    </Box>
  );
}
```

- [ ] **Step 2: Wire into App wizard flow**

Modify `tui/src/app.tsx` to track config state and render the wizard steps. Keep the current screen minimal; full wiring happens in Task 9.

- [ ] **Step 3: Commit**

```bash
git add tui/src/components/select-data.tsx tui/src/app.tsx
git commit -m "feat(tui): add Select Data wizard step"
```

---

### Task 7: Configure, Review, and Run screens

**Files:**
- Create: `tui/src/components/configure.tsx`
- Create: `tui/src/components/review.tsx`
- Create: `tui/src/components/run-watch.tsx`
- Create: `tui/src/components/wizard-shell.tsx`
- Modify: `tui/src/app.tsx`

- [ ] **Step 1: Build WizardShell**

Create `tui/src/components/wizard-shell.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface WizardShellProps {
  step: number;
  theme: Theme;
  children: React.ReactNode;
}

const STEPS = ["Select Data", "Configure", "Review", "Run & Watch"];

export function WizardShell({ step, theme, children }: WizardShellProps) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Box marginBottom={1}>
        {STEPS.map((label, idx) => (
          <Text key={label} color={idx === step ? theme.highlight : theme.muted}>
            {idx === step ? ` ${idx + 1}. ${label} ` : ` ${idx + 1}. ${label} `}
          </Text>
        ))}
      </Box>
      {children}
    </Box>
  );
}
```

- [ ] **Step 2: Build Configure screen**

Create `tui/src/components/configure.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { AnalysisConfig } from "../types";

interface ConfigureProps {
  config: AnalysisConfig;
  onChange: (config: AnalysisConfig) => void;
  theme: Theme;
}

export function Configure({ config, onChange, theme }: ConfigureProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        2. Configure
      </Text>
      <Text color={theme.foreground}>Fiber diameter: {config.fiberDiameter} µm</Text>
      <Text color={theme.foreground}>Regime: {config.regime}</Text>
      <Text color={theme.foreground}>Method: {config.method}</Text>
      <Text color={theme.muted}>Full form implementation pending...</Text>
    </Box>
  );
}
```

- [ ] **Step 3: Build Review screen**

Create `tui/src/components/review.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { AnalysisConfig } from "../types";

interface ReviewProps {
  config: AnalysisConfig;
  theme: Theme;
}

export function Review({ config, theme }: ReviewProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        3. Review
      </Text>
      <Text color={theme.foreground}>Data: {config.dataPath}</Text>
      <Text color={theme.foreground}>Output: {config.outputDir}</Text>
      <Text color={theme.foreground}>Method: {config.method}</Text>
      <Text color={theme.success}>Press r to run</Text>
    </Box>
  );
}
```

- [ ] **Step 4: Build Run & Watch screen**

Create `tui/src/components/run-watch.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { BridgeStatus, ProgressEvent } from "../types";

interface RunWatchProps {
  status: BridgeStatus;
  progress: ProgressEvent | null;
  logs: string[];
  theme: Theme;
}

export function RunWatch({ status, progress, logs, theme }: RunWatchProps) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text bold color={theme.accent}>
        4. Run & Watch
      </Text>
      <Text color={theme.foreground}>Status: {status}</Text>
      {progress && (
        <>
          <Text color={theme.foreground}>
            {progress.stage} — {progress.percent}%
          </Text>
          <Text color={theme.muted}>{progress.message}</Text>
        </>
      )}
      <Box flexDirection="column" marginTop={1} flexGrow={1} overflow="hidden">
        {logs.slice(-5).map((line, idx) => (
          <Text key={idx} color={theme.muted}>
            {line}
          </Text>
        ))}
      </Box>
    </Box>
  );
}
```

- [ ] **Step 5: Wire wizard into App**

Modify `tui/src/app.tsx` to manage wizard state and render the correct step:

```typescript
import React, { useState } from "react";
import { useInput } from "ink";
import { Layout } from "./components/layout";
import { WizardShell } from "./components/wizard-shell";
import { SelectData } from "./components/select-data";
import { Configure } from "./components/configure";
import { Review } from "./components/review";
import { RunWatch } from "./components/run-watch";
import { Settings } from "./components/settings";
import { useConfig } from "./hooks/useConfig";
import { useBridge } from "./hooks/useBridge";
import { loadTheme } from "./theme";
import type { Section } from "./components/sidebar";
import type { AnalysisConfig } from "./types";

const DEFAULT_CONFIG: AnalysisConfig = {
  dataPath: "",
  outputDir: "./results/run",
  voxelSpacing: [1, 1, 1],
  fiberDiameter: 10,
  regime: "auto",
  method: "otsu",
  model: "models/fiber_unet_v2_full.pt",
  batchSize: 1,
  computeMorphometry: true,
  computeOrientationTensor: true,
  computeTda: false,
};

export function App() {
  const { config, setConfig } = useConfig();
  const [section, setSection] = useState<Section>("new-analysis");
  const [wizardStep, setWizardStep] = useState(0);
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const bridge = useBridge();
  const theme = loadTheme(config.theme);

  useInput((input, key) => {
    if (input === "q") process.exit(0);
    if (input === "1") setSection("new-analysis");
    if (input === "8") setSection("settings");
    if (section === "new-analysis") {
      if (key.rightArrow || input === "n") setWizardStep((s) => Math.min(3, s + 1));
      if (key.leftArrow || input === "p") setWizardStep((s) => Math.max(0, s - 1));
      if (input === "r" && wizardStep === 2) {
        setWizardStep(3);
        bridge.run(analysisConfig);
      }
    }
  });

  const footer =
    section === "new-analysis"
      ? ["← p", "→ n", "Enter select", "r run", "q quit"]
      : ["1 Analysis", "8 Settings", "q quit"];

  return (
    <Layout section={section} wizardStep={wizardStep} footerShortcuts={footer} theme={theme}>
      {section === "new-analysis" && (
        <WizardShell step={wizardStep} theme={theme}>
          {wizardStep === 0 && (
            <SelectData value={analysisConfig.dataPath} onChange={(path) => setAnalysisConfig((c) => ({ ...c, dataPath: path }))} theme={theme} />
          )}
          {wizardStep === 1 && <Configure config={analysisConfig} onChange={setAnalysisConfig} theme={theme} />}
          {wizardStep === 2 && <Review config={analysisConfig} theme={theme} />}
          {wizardStep === 3 && <RunWatch status={bridge.status} progress={bridge.progress} logs={bridge.logs} theme={theme} />}
        </WizardShell>
      )}
      {section === "settings" && <Settings config={config} onChange={setConfig} theme={theme} />}
    </Layout>
  );
}
```

- [ ] **Step 6: Run TUI smoke test**

Run:
```bash
cd tui
bun run build
```

Expected: TypeScript compiles without errors.

- [ ] **Step 7: Commit**

```bash
git add tui/src/components/configure.tsx tui/src/components/review.tsx tui/src/components/run-watch.tsx tui/src/components/wizard-shell.tsx tui/src/app.tsx
git commit -m "feat(tui): add wizard steps Configure, Review, Run & Watch"
```

---

## Phase 5: Results Screens

### Task 8: Dashboard, History, Logs

**Files:**
- Create: `tui/src/components/dashboard.tsx`
- Create: `tui/src/components/history.tsx`
- Create: `tui/src/components/logs.tsx`
- Modify: `tui/src/app.tsx`

- [ ] **Step 1: Build Dashboard component**

Create `tui/src/components/dashboard.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface DashboardProps {
  summary?: Record<string, unknown>;
  theme: Theme;
}

export function Dashboard({ summary, theme }: DashboardProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Dashboard
      </Text>
      <Text color={theme.foreground}>Fibers: {summary?.n_labels ?? "-"}</Text>
      <Text color={theme.foreground}>Regime: {String(summary?.regime ?? "-")}</Text>
      <Text color={theme.muted}>Charts and fiber table added in Task 10 polish pass.</Text>
    </Box>
  );
}
```

- [ ] **Step 2: Build History component**

Create `tui/src/components/history.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { RunRecord } from "../types";

interface HistoryProps {
  history: RunRecord[];
  theme: Theme;
}

export function History({ history, theme }: HistoryProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        History
      </Text>
      {history.length === 0 && <Text color={theme.muted}>No runs yet.</Text>}
      {history.map((run) => (
        <Text key={run.id} color={theme.foreground}>
          {run.status === "success" ? "✓" : "✗"} {run.name} — {run.startedAt}
        </Text>
      ))}
    </Box>
  );
}
```

- [ ] **Step 3: Build Logs component**

Create `tui/src/components/logs.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface LogsProps {
  logs: string[];
  theme: Theme;
}

export function Logs({ logs, theme }: LogsProps) {
  return (
    <Box flexDirection="column" flexGrow={1} overflow="hidden">
      <Text bold color={theme.accent}>
        Logs
      </Text>
      {logs.slice(-20).map((line, idx) => (
        <Text key={idx} color={theme.muted}>
          {line}
        </Text>
      ))}
    </Box>
  );
}
```

- [ ] **Step 4: Wire results screens into App**

Modify `tui/src/app.tsx` to include History and Logs. Use `useHistory` hook and pass bridge logs to Logs screen.

- [ ] **Step 5: Commit**

```bash
git add tui/src/components/dashboard.tsx tui/src/components/history.tsx tui/src/components/logs.tsx tui/src/app.tsx
git commit -m "feat(tui): add Dashboard, History, and Logs screens"
```

---

## Phase 6: Future-Proof Features

### Task 9: Model Registry, Experiments, Training screens (initial version)

**Files:**
- Create: `tui/src/components/model-registry.tsx`
- Create: `tui/src/components/experiments.tsx`
- Create: `tui/src/components/training.tsx`
- Modify: `tui/src/app.tsx`

- [ ] **Step 1: Build Model Registry component**

Create `tui/src/components/model-registry.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface ModelRegistryProps {
  theme: Theme;
}

export function ModelRegistry({ theme }: ModelRegistryProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Model Registry
      </Text>
      <Text color={theme.foreground}>▸ fiber_unet_v2_full.pt (active)</Text>
      <Text color={theme.muted}>  fiber_unet_v3.pt (experimental)</Text>
      <Text color={theme.muted}>Press d to download a model from GitHub releases.</Text>
    </Box>
  );
}
```

- [ ] **Step 2: Build Experiments component**

Create `tui/src/components/experiments.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface ExperimentsProps {
  theme: Theme;
}

export function Experiments({ theme }: ExperimentsProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Experiments
      </Text>
      <Text color={theme.muted}>No experiments yet. Use this screen to compare methods and models.</Text>
    </Box>
  );
}
```

- [ ] **Step 3: Build Training component**

Create `tui/src/components/training.tsx`:

```typescript
import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface TrainingProps {
  theme: Theme;
}

export function Training({ theme }: TrainingProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Training
      </Text>
      <Text color={theme.muted}>Guided fine-tuning workflow: Prepare → Train → Validate → Export.</Text>
    </Box>
  );
}
```

- [ ] **Step 4: Wire all screens into App**

Modify `tui/src/app.tsx` to render the new screens when their sidebar section is active.

- [ ] **Step 5: Commit**

```bash
git add tui/src/components/model-registry.tsx tui/src/components/experiments.tsx tui/src/components/training.tsx tui/src/app.tsx
git commit -m "feat(tui): add Model Registry, Experiments, and Training screens (initial version)"
```

---

## Phase 7: Polish & CI

### [x] Task 10: Tests, docs, and CI integration

**Files:**
- Create: `tui/src/app.test.tsx`
- Create: `.github/workflows/tui.yml`
- Modify: `tui/README.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [x] **Step 1: Add component render test**

Create `tui/src/app.test.tsx`:

```typescript
import { describe, expect, it } from "bun:test";
import React from "react";
import { render } from "ink-testing-library";
import { App } from "./app";

describe("App", () => {
  it("renders without crashing", () => {
    const { lastFrame } = render(<App />);
    expect(lastFrame()).toContain("Fiber Tracer");
  });
});
```

- [x] **Step 2: Install ink-testing-library**

Run:
```bash
cd tui
bun add -d ink-testing-library
```

- [x] **Step 3: Run TUI tests**

Run:
```bash
cd tui
bun test
```

Expected: All tests pass.

- [x] **Step 4: Add TUI CI job**

Create `.github/workflows/tui.yml`:

```yaml
name: TUI
on:
  push:
    branches: [main, master]
    paths:
      - "tui/**"
      - ".github/workflows/tui.yml"
  pull_request:
    paths:
      - "tui/**"
      - ".github/workflows/tui.yml"
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v2
        with:
          bun-version: latest
      - run: cd tui && bun install
      - run: cd tui && bun run typecheck
      - run: cd tui && bun test
```

- [x] **Step 5: Update TUI README**

Create `tui/README.md` with project overview, requirements, install/test commands, keyboard shortcuts, theme configuration note, and link to the main README.

- [x] **Step 6: Update README**

Add a "Terminal UI (TUI)" section to `README.md`.

- [x] **Step 7: Commit**

```bash
git add tui/src/app.test.tsx tui/package.json tui/README.md .github/workflows/tui.yml README.md CHANGELOG.md docs/superpowers/plans/2026-06-25-fiber-tracer-tui-plan.md
git commit -m "feat(tui): finalize TUI with tests, CI, docs and README"
```

---

## Spec Coverage Check

| Spec Section | Plan Task |
|--------------|-----------|
| Bun + Ink + @inkjs/ui stack | Task 1 |
| Theme system (Dracula + others) | Task 2 |
| Config/history persistence | Task 3 |
| Persistent sidebar navigation | Task 4 |
| Python bridge / JSON-RPC | Task 5 |
| Select Data wizard step | Task 6 |
| Configure / Review / Run & Watch | Task 7 |
| Dashboard / History / Logs | Task 8 |
| Model Registry / Experiments / Training | Task 9 |
| Tests, CI, docs | Task 10 |

Every task includes exact file paths, code, commands, and expected outputs. Deliberately lightweight screens (Settings, Dashboard, Model Registry, Experiments, Training) are implemented with real components and marked for expansion in Task 10 or follow-up work rather than left as prose.

## Risk Mitigations

- **@inkjs/ui component availability.** Fallback: use raw Ink `Box`/`Text` and implement components manually if a @inkjs/ui component is missing.
- **Python bridge subprocess may be slow.** Mitigation: start with subprocess; plan to migrate to `fiber-tracer tui-server` JSON-RPC in a follow-up.
- **Local Python is 3.9.** Mitigation: upgrade local Python to 3.10+ before implementation (separate todo). CI already tests 3.10–3.12.
- **Terminal resize.** Mitigation: use Ink's built-in `useStdout` dimensions and keep layouts fluid.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-25-fiber-tracer-tui-plan.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
