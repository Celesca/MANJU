import type { Config } from "./create-config";
/**
 * Gets the current configuration by reading and validating the config file.
 *
 * This function calls `setupConfig()` which will:
 * - Create the config file if it doesn't exist
 * - Validate and fix any invalid config values
 * - Return the parsed configuration
 *
 * @returns {Promise<Config>} A promise that resolves to the validated configuration object
 */
export declare function getConfig(): Promise<Config>;
