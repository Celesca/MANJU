/**
 * Sets up the register script in the project's package.json file.
 *
 * This function checks if the postinstall script already exists in the package.json file.
 * If it does not exist, it adds the register command to the postinstall script.
 * If it does exist, it appends the register command to the existing postinstall script.
 */
export declare function setupRegister(): Promise<void>;
