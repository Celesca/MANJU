/**
 * Safely retrieves a value from a nested object using a dot-notated path string
 *
 * @template T - The type of the input object
 * @param {T} input - The input object to traverse
 * @param {string} path - A dot-notated string path (e.g. "user.address.street")
 * @returns {any} The value at the specified path, or undefined if the path is invalid
 */
export declare function get<T>(input: T, path: string): any;
