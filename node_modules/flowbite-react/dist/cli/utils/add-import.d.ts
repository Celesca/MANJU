/**
 * Adds an import statement to a given content string.
 *
 * @param {Object} options - The options for adding the import.
 * @param {string} options.content - The content string to modify.
 * @param {string} options.importPath - The path of the module to import.
 * @param {string} options.importName - The name to assign to the imported module.
 * @returns {string} The modified content string with the import added.
 */
export declare function addImport({ content, importPath, importName, }: {
    content: string;
    importPath: string;
    importName: string;
}): string;
