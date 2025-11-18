import { type ComponentProps, type ReactNode } from "react";
import type { ThemingProps } from "../../types";
import type { ClipboardWithIconTheme } from "./ClipboardWithIcon";
import type { ClipboardWithIconTextTheme } from "./ClipboardWithIconText";
export interface ClipboardTheme {
    button: {
        base: string;
        label: string;
    };
    withIcon: ClipboardWithIconTheme;
    withIconText: ClipboardWithIconTextTheme;
}
export interface ClipboardProps extends ComponentProps<"button">, ThemingProps<ClipboardTheme["button"]> {
    valueToCopy: string;
    label?: ReactNode;
}
export declare const Clipboard: import("react").ForwardRefExoticComponent<Omit<ClipboardProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
