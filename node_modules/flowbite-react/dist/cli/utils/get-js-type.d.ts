/**
 * Determines the type of JavaScript module based on its content.
 *
 * @param {string} content - The content of the JavaScript module.
 * @returns {{ isCJS: boolean; isESM: boolean }} An object indicating whether the content is a CommonJS (CJS) or an ECMAScript Module (ESM).
 */
export declare function getJsType(content: string): {
    isCJS: boolean;
    isESM: boolean;
};
