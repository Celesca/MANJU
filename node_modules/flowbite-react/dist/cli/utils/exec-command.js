import { spawn } from 'child_process';

function execCommand(command, args = [], options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, options);
    let stdout = "";
    let stderr = "";
    if (child.stdout) {
      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
    }
    if (child.stderr) {
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });
    }
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr, exitCode: code || 0 });
      } else {
        reject(new Error(`Process exited with code ${code}: ${stderr}`));
      }
    });
    child.on("error", (err) => {
      reject(err);
    });
  });
}

export { execCommand };
//# sourceMappingURL=exec-command.js.map
