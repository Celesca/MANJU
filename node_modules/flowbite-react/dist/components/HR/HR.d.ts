import type { ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { HRIconTheme } from "./HRIcon";
import type { HRSquareTheme } from "./HRSquare";
import type { HRTextTheme } from "./HRText";
import type { HRTrimmedTheme } from "./HRTrimmed";
export interface HRTheme {
    root: HRRootTheme;
    trimmed: HRTrimmedTheme;
    icon: HRIconTheme;
    text: HRTextTheme;
    square: HRSquareTheme;
}
export interface HRRootTheme {
    base: string;
}
export interface HRProps extends Omit<ComponentProps<"hr">, "ref">, ThemingProps<HRRootTheme> {
}
export declare const HR: import("react").ForwardRefExoticComponent<HRProps & import("react").RefAttributes<HTMLHRElement>>;
