const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const rootDir = path.resolve(__dirname, "..");
const backendDir = path.join(rootDir, "backend");

const windowsPython = path.join(backendDir, "venv", "Scripts", "python.exe");
const unixPython = path.join(backendDir, "venv", "bin", "python");

const candidates = process.platform === "win32"
  ? [windowsPython, "py", "python"]
  : [unixPython, "python3", "python"];

const args = [
  "-m",
  "uvicorn",
  "app.main:app",
  "--reload",
  "--host",
  "0.0.0.0",
  "--port",
  "8000",
];

const backendPort = 8000;

function isFile(filePath) {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function hasUvicorn(command) {
  const probe = spawnSync(command, ["-c", "import uvicorn"], {
    cwd: backendDir,
    stdio: "ignore",
  });
  return probe.status === 0;
}

function resolvePython() {
  for (const command of candidates) {
    if (path.isAbsolute(command) && !isFile(command)) {
      continue;
    }
    if (hasUvicorn(command)) {
      return command;
    }
  }
  return null;
}

const pythonCommand = resolvePython();

if (!pythonCommand) {
  console.error("Could not find a Python interpreter with uvicorn installed.");
  console.error("Run backend setup first: cd backend && python -m pip install -r requirements.txt");
  process.exit(1);
}

function ensurePortAvailable(port) {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.once("error", (error) => {
      if (error && error.code === "EADDRINUSE") {
        reject(new Error(`Port ${port} is already in use.`));
        return;
      }
      reject(error);
    });
    probe.once("listening", () => {
      probe.close(() => resolve());
    });
    probe.listen(port, "0.0.0.0");
  });
}

async function startBackend() {
  try {
    await ensurePortAvailable(backendPort);
  } catch (error) {
    console.error(error.message || `Backend port ${backendPort} is unavailable.`);
    console.error("Stop the process using this port and run npm run dev again.");
    process.exit(1);
  }

  const child = spawn(pythonCommand, args, {
    cwd: backendDir,
    stdio: "inherit",
  });

  child.on("error", (error) => {
    if (error && error.code === "ENOENT") {
      console.error("Python interpreter not found:", pythonCommand);
    }
    process.exit(1);
  });

  child.on("exit", (code) => {
    process.exit(code ?? 1);
  });
}

startBackend();
