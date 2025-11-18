import type { DeepPartialApplyTheme, DeepPartialBoolean } from "../types";
/**
 * Memoize wrapper around `resolveTheme` function.
 *
 * @param {...Parameters<typeof resolveTheme>} input - Arguments to pass to `resolveTheme` function
 * @returns {ReturnType<typeof resolveTheme>} The resolved theme configuration
 */
export declare function useResolveTheme<T>(...input: Parameters<typeof resolveTheme<T>>): ReturnType<typeof resolveTheme<T>>;
/**
 * A custom React hook that memoizes a value similar to `useMemo`, but with stable dependency comparison.
 * This hook ensures that the memoized value only updates when the dependencies have actually changed,
 * using deep equality comparison instead of reference equality.
 *
 * @template T - The type of the memoized value
 * @param {() => T} factory - A function that creates the value to be memoized
 * @param {unknown[]} dependencies - An array of dependencies that determine when the value should be recalculated
 * @returns {T} The memoized value that only changes when dependencies change (using deep equality)
 */
export declare function useStableMemo<T>(factory: () => T, dependencies: unknown[]): T;
/**
 * Adds prefix to `base` and merges with custom themes, applying optional `clearTheme` and `applyTheme` modifications.
 *
 * @template T - The type of the base theme.
 * @param {[base, ...custom[]]} themes - An array where the first element is the base theme and the rest are custom themes.
 * @param {DeepPartialBoolean<T>[]} clearThemeList - An array of `clearTheme` modifications to apply to the base theme.
 * @param {DeepPartialApplyTheme<T>[]} applyThemeList - An optional array of `applyTheme` modifications to apply to the merged theme.
 * @returns {T} - The resolved and merged theme.
 */
export declare function resolveTheme<T>([base, ...custom]: [
    /** base theme */
    T,
    /** custom themes */
    ...unknown[]
], clearThemeList?: DeepPartialBoolean<T>[], applyThemeList?: DeepPartialApplyTheme<T>[]): T;
