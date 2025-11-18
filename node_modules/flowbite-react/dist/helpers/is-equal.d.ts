/**
 * Performs a deep equality comparison between two values.
 *
 * This function recursively compares objects and arrays, handling special cases like:
 * - RegExp objects (comparing source and flags)
 * - Objects with custom valueOf or toString methods
 * - Arrays (comparing length and each element)
 * - NaN values (which are considered equal to each other)
 *
 * @param {any} a - The first value to compare
 * @param {any} b - The second value to compare
 * @returns {boolean} True if the values are deeply equal, false otherwise
 *
 * @example
 * ```ts
 * isEqual({a: 1, b: 2}, {a: 1, b: 2}); // true
 * isEqual([1, 2, 3], [1, 2, 3]); // true
 * isEqual(/abc/g, /abc/g); // true
 * isEqual(NaN, NaN); // true
 * ```
 */
export declare function isEqual(a: any, b: any): boolean;
