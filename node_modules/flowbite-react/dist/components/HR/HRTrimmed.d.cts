import type { ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface HRTrimmedTheme {
    base: string;
}
export interface HRTrimmedProps extends Omit<ComponentProps<"hr">, "ref">, ThemingProps<HRTrimmedTheme> {
}
export declare const HRTrimmed: import("react").ForwardRefExoticComponent<HRTrimmedProps & import("react").RefAttributes<HTMLHRElement>>;
