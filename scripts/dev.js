const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const readline = require("node:readline");

const rootDir = path.resolve(__dirname, "..");
const backendDir = path.join(rootDir, "backend");
const frontendDir = path.join(rootDir, "frontend");

const windowsPython = path.join(backendDir, "venv", "Scripts", "python.exe");
const unixPython = path.join(backendDir, "venv", "bin", "python");

const pythonCandidates = process.platform === "win32"
  ? [windowsPython, "py", "python"]
  : [unixPython, "python3", "python"];

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

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
  for (const command of pythonCandidates) {
    if (path.isAbsolute(command) && !isFile(command)) {
      continue;
    }
    if (hasUvicorn(command)) {
      return command;
    }
  }
  return null;
}

function canBind(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "0.0.0.0");
  });
}

async function findOpenPort(startPort, attempts = 50) {
  for (let port = startPort; port < startPort + attempts; port += 1) {
    // eslint-disable-next-line no-await-in-loop
    if (await canBind(port)) {
      return port;
    }
  }
  return null;
}

function pipeWithPrefix(stream, prefix) {
  if (!stream) return;
  const rl = readline.createInterface({ input: stream });
  rl.on("line", (line) => {
    console.log(`[${prefix}] ${line}`);
  });
}

async function main() {
  const pythonCommand = resolvePython();
  if (!pythonCommand) {
    console.error("[BACKEND] Could not find a Python interpreter with uvicorn installed.");
    console.error("[BACKEND] Run backend setup first: cd backend && python -m pip install -r requirements.txt");
    process.exit(1);
  }

  const backendPort = await findOpenPort(8000);
  if (!backendPort) {
    console.error("[BACKEND] Could not find an available port in range 8000-8049.");
    process.exit(1);
  }

  const frontendPort = await findOpenPort(3000);
  if (!frontendPort) {
    console.error("[FRONTEND] Could not find an available port in range 3000-3049.");
    process.exit(1);
  }

  console.log(`[DEV] Backend port: ${backendPort}`);
  console.log(`[DEV] Frontend port: ${frontendPort}`);
  console.log(`[DEV] API URL for frontend: http://localhost:${backendPort}`);

  const backendArgs = [
    "-m",
    "uvicorn",
    "app.main:app",
    "--reload",
    "--reload-dir",
    "app",
    "--host",
    "0.0.0.0",
    "--port",
    String(backendPort),
  ];

  const backend = spawn(pythonCommand, backendArgs, {
    cwd: backendDir,
    stdio: ["inherit", "pipe", "pipe"],
    env: process.env,
  });

  const frontend = spawn(npmCommand, ["run", "dev", "--", "--port", String(frontendPort)], {
    cwd: frontendDir,
    stdio: ["inherit", "pipe", "pipe"],
    env: {
      ...process.env,
      NEXT_PUBLIC_API_URL: `http://localhost:${backendPort}`,
      API_URL: `http://localhost:${backendPort}`,
    },
  });

  pipeWithPrefix(backend.stdout, "BACKEND");
  pipeWithPrefix(backend.stderr, "BACKEND");
  pipeWithPrefix(frontend.stdout, "FRONTEND");
  pipeWithPrefix(frontend.stderr, "FRONTEND");

  let shuttingDown = false;

  function shutdown(code = 0) {
    if (shuttingDown) return;
    shuttingDown = true;

    if (!backend.killed) backend.kill("SIGTERM");
    if (!frontend.killed) frontend.kill("SIGTERM");

    setTimeout(() => process.exit(code), 300);
  }

  backend.on("error", () => shutdown(1));
  frontend.on("error", () => shutdown(1));

  backend.on("exit", (code) => {
    if (!shuttingDown) {
      console.error(`[BACKEND] exited with code ${code ?? 1}`);
      shutdown(code ?? 1);
    }
  });

  frontend.on("exit", (code) => {
    if (!shuttingDown) {
      console.error(`[FRONTEND] exited with code ${code ?? 1}`);
      shutdown(code ?? 1);
    }
  });

  process.on("SIGINT", () => shutdown(0));
  process.on("SIGTERM", () => shutdown(0));
}

main().catch((err) => {
  console.error("[DEV] Startup failed:", err?.message || err);
  process.exit(1);
});
