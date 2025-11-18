import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface ClipboardWithIconTextTheme {
    base: string;
    label: {
        base: string;
        defaultText: string;
        successText: string;
    };
    icon: {
        defaultIcon: string;
        successIcon: string;
    };
}
export interface ClipboardWithIconTextProps extends ComponentProps<"button">, ThemingProps<ClipboardWithIconTextTheme> {
    valueToCopy: string;
    label?: string;
    icon?: FC<ComponentProps<"svg">>;
}
export declare const ClipboardWithIconText: import("react").ForwardRefExoticComponent<Omit<ClipboardWithIconTextProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
