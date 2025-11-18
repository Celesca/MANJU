import type { ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface HRTextTheme {
    base: string;
    hrLine: string;
    text: string;
}
export interface HRTextProps extends Omit<ComponentProps<"hr">, "ref">, ThemingProps<HRTextTheme> {
    text: string;
}
export declare const HRText: import("react").ForwardRefExoticComponent<HRTextProps & import("react").RefAttributes<HTMLHRElement>>;
