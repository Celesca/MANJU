import type React from "react";
import type { ThemeMode } from "../hooks/use-theme-mode";
export interface ThemeModeScriptProps extends React.ComponentPropsWithoutRef<"script"> {
    /**
     * The current theme mode to use
     *
     * @type {"light" | "dark" | "auto"}
     */
    mode?: ThemeMode;
    /**
     * The default theme mode if none is set
     *
     * @type {"light" | "dark" | "auto"}
     * @default "auto"
     */
    defaultMode?: ThemeMode;
    /**
     * Key used to store theme mode in localStorage
     *
     * @type {string}
     * @default "flowbite-theme-mode"
     */
    localStorageKey?: string;
    /**
     * The prefix to use for the theme mode class
     *
     * @type {string}
     * @default ""
     */
    prefix?: string;
    /**
     * The version of Tailwind CSS to use
     *
     * @type {3 | 4}
     * @default 4
     */
    version?: 3 | 4;
}
/**
 * A script component that handles theme mode initialization
 *
 * @param {Object} props - The component props
 * @param {ThemeMode} [props.mode] - The current theme mode to use
 * @param {ThemeMode} [props.defaultMode="auto"] - The default theme mode if none is set
 * @param {string} [props.localStorageKey="flowbite-theme-mode"] - Key used to store theme mode in localStorage
 * @param {React.ComponentPropsWithoutRef<"script">} props.others - Additional script element props
 * @returns {JSX.Element} Script element that initializes theme mode
 */
export declare function ThemeModeScript({ mode, defaultMode, localStorageKey, prefix, version, ...others }: ThemeModeScriptProps): JSX.Element;
export declare namespace ThemeModeScript {
    var displayName: string;
}
/**
 * Generates a script string that handles theme mode initialization
 *
 * @param {Object} options - The options object
 * @param {ThemeMode} [options.mode] - The current theme mode to use
 * @param {ThemeMode} [options.defaultMode="auto"] - The default theme mode if none is set
 * @param {string} [options.localStorageKey="flowbite-theme-mode"] - Key used to store theme mode in localStorage
 * @param {string} [options.prefix=""] - The prefix to use for the theme mode class
 * @returns {string} Script string that initializes theme mode
 */
export declare function getThemeModeScript(props?: {
    mode?: ThemeMode;
    defaultMode?: ThemeMode;
    localStorageKey?: string;
    prefix?: string;
    version?: 3 | 4;
}): string;
/**
 * Initializes the theme mode by checking localStorage, provided mode, or default mode
 * and applies the appropriate class to the document element
 *
 * @param {Object} options - The options object
 * @param {ThemeMode} [options.mode] - The current theme mode to use
 * @param {ThemeMode} [options.defaultMode="auto"] - The default theme mode if none is set
 * @param {string} [options.localStorageKey="flowbite-theme-mode"] - Key used to store theme mode in localStorage
 * @param {string} [options.prefix=""] - The prefix to use for the theme mode class
 * @param {3 | 4} [options.version=4] - The version of Tailwind CSS to use
 * @returns {void}
 */
export declare function initThemeMode(props?: {
    mode?: ThemeMode;
    defaultMode?: ThemeMode;
    localStorageKey?: string;
    prefix?: string;
    version?: 3 | 4;
}): void;
