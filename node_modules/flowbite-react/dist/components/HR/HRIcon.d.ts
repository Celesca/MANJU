import type { ComponentProps, FC } from "react";
import type { ThemingProps } from "../../types";
export interface HRIconTheme {
    base: string;
    hrLine: string;
    icon: {
        base: string;
        icon: string;
    };
}
export interface HRIconProps extends Omit<ComponentProps<"hr">, "ref">, ThemingProps<HRIconTheme> {
    icon?: FC<ComponentProps<"svg">>;
}
export declare const HRIcon: import("react").ForwardRefExoticComponent<HRIconProps & import("react").RefAttributes<HTMLHRElement>>;
