/**
 * Reads the class list file and returns its content as an array of strings.
 *
 * This function attempts to read the file specified by `classListFilePath`, parse its content as JSON,
 * and return the parsed array. If the file cannot be read or parsed, an empty array is returned.
 *
 * @returns {Promise<string[]>} A promise that resolves to an array of strings representing the class list.
 */
export declare function getClassList(): Promise<string[]>;
