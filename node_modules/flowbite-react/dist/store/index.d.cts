import type { ThemeMode } from "../hooks/use-theme-mode";
import type { DeepPartial } from "../types";
export type StoreProps = DeepPartial<{
    /**
     * Whether to generate dark mode styles
     *
     * @default true
     */
    dark: boolean;
    /**
     * Theme mode
     *
     * Can be `"light"`, `"dark"`, or `"auto"`
     *
     * @default "auto"
     */
    mode: ThemeMode;
    /**
     * Prefix to apply to base class list
     *
     * @default undefined
     */
    prefix: string;
    /**
     * The version of Tailwind CSS to use
     *
     * @default 4
     */
    version: 3 | 4;
}>;
export declare function setStore(data: StoreProps): void;
export declare function getDark(): StoreProps["dark"];
export declare function getMode(): StoreProps["mode"];
export declare function getPrefix(): StoreProps["prefix"];
export declare function getVersion(): StoreProps["version"];
