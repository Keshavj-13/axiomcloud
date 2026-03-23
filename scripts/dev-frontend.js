const { spawn } = require("node:child_process");
const net = require("node:net");
const path = require("node:path");

const rootDir = path.resolve(__dirname, "..");
const frontendDir = path.join(rootDir, "frontend");
const frontendPort = 3000;

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const args = ["run", "dev", "--", "--port", String(frontendPort)];

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

async function startFrontend() {
  try {
    await ensurePortAvailable(frontendPort);
  } catch (error) {
    console.error(error.message || `Frontend port ${frontendPort} is unavailable.`);
    console.error("Stop the process using this port and run npm run dev again.");
    process.exit(1);
  }

  const child = spawn(npmCommand, args, {
    cwd: frontendDir,
    stdio: "inherit",
  });

  child.on("error", () => {
    console.error("Failed to start frontend dev server.");
    process.exit(1);
  });

  child.on("exit", (code) => {
    process.exit(code ?? 1);
  });
}

startFrontend();
