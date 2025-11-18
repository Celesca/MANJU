/**
 * Gets the package.json contents for a given package.
 *
 * @param packageName - The name of the package to get the package.json from
 * @returns The contents of the package.json file or null if the package is not installed
 */
export declare function getModulePackageJson<T extends {
    name: string;
    version: string;
} & Record<string, unknown>>(packageName: string): Promise<T | null>;
