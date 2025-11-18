import { type Config } from "./create-config";
/**
 * Creates a logger to track and warn about `<ThemeInit />` component usage.
 *
 * @param {Config} config - The configuration object used to check
 */
export declare function createInitLogger(config: Config): {
    config: Config;
    checkedMap: Map<string, boolean>;
    readonly isCustomConfig: boolean;
    readonly showWarning: boolean;
    /**
     * Checks if `<ThemeInit />` component is used in the given file content
     *
     * @param path - The path to the file being checked
     * @param content - The file content to search in
     */
    check(path: string, content: string): void;
    /**
     * Logs a warning if `<ThemeInit />` component is not used in the project and the configuration `dark`, `prefix` or `version` differs from default values.
     */
    log(): void;
};
/**
 * Checks if `<ThemeInit />` component is used in the given file content
 *
 * @param content - The file content to search in
 * @returns boolean indicating if ThemeInit is used
 */
export declare function hasThemeInit(content: string): boolean;
