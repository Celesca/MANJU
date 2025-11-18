export interface FindFilesOptions {
    patterns: string[];
    startDir?: string;
    excludeDirs?: string[];
    parallel?: boolean;
    recursive?: boolean;
}
/**
 * Finds files matching specified patterns within a directory.
 *
 * This function recursively searches a directory for files that match the given patterns.
 * It can be configured to search in parallel or sequentially, and to exclude specific directories.
 *
 * @param {FindFilesOptions} options - The options for the file search.
 * @param {string[]} options.patterns - The patterns to match files against.
 * @param {string} [options.startDir="."] - The directory to start the search from.
 * @param {string[]} [options.excludeDirs=[]] - Directories to exclude from the search.
 * @param {boolean} [options.parallel=true] - Whether to search in parallel or sequentially.
 * @param {boolean} [options.recursive=true] - Whether to search recursively.
 * @returns {Promise<string[]>} A promise that resolves to an array of paths of files that match the patterns.
 */
export declare function findFiles({ patterns, startDir, excludeDirs, parallel, recursive, }: FindFilesOptions): Promise<string[]>;
