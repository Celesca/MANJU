/**
 * Builds a list of CSS classes based on the provided components and prefix.
 *
 * @param {Object} options - The options for building the class list.
 * @param {string[]} options.components - The components to include in the class list.
 * @param {boolean} options.dark - Whether to include dark mode classes.
 * @param {string} options.prefix - The prefix to apply to the class names.
 * @param {3 | 4} options.version - The version of Tailwind CSS to use.
 * @returns {string[]} The sorted list of CSS classes.
 */
export declare function buildClassList({ components, dark, prefix, version, }: {
    components: string[];
    dark: boolean;
    prefix: string;
    version: 3 | 4;
}): string[];
