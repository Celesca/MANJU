import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface ClipboardWithIconTheme {
    base: string;
    icon: {
        defaultIcon: string;
        successIcon: string;
    };
}
export interface ClipboardWithIconProps extends ComponentProps<"button">, ThemingProps<ClipboardWithIconTheme> {
    valueToCopy: string;
    icon?: FC<ComponentProps<"svg">>;
}
export declare const ClipboardWithIcon: import("react").ForwardRefExoticComponent<Omit<ClipboardWithIconProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
