import { type SpawnOptions } from "child_process";
/**
 * Runs a command asynchronously and captures its output.
 *
 * @param {string} command - The command to execute (e.g., "npm").
 * @param {string[]} [args=[]] - Arguments for the command (e.g., ["install", "package-name"]).
 * @param {SpawnOptions} [options={}] - Optional spawn configuration options.
 * @returns {Promise<{ stdout: string; stderr: string; exitCode: number }>} - Resolves with command output and exit code.
 * @throws {Error} If the process exits with a non-zero code.
 */
export declare function execCommand(command: string, args?: string[], options?: SpawnOptions): Promise<{
    stdout: string;
    stderr: string;
    exitCode: number;
}>;
