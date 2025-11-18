/**
 * Normalizes a path for use in import statements, ensuring forward slashes are used
 * regardless of the operating system.
 *
 * @param importPath - The path to normalize
 * @returns The normalized path with forward slashes
 */
export declare function normalizeImportPath(importPath: string): string;
/**
 * Joins path segments and normalizes the result to use forward slashes.
 * This is useful for paths that need to be used in import statements or URLs.
 *
 * @param segments - Path segments to join
 * @returns The normalized path with forward slashes
 */
export declare function joinNormalizedPath(...segments: string[]): string;
