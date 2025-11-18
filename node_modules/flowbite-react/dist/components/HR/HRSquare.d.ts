import type { ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface HRSquareTheme {
    base: string;
}
export interface HRSquareProps extends Omit<ComponentProps<"hr">, "ref">, ThemingProps<HRSquareTheme> {
}
export declare const HRSquare: import("react").ForwardRefExoticComponent<HRSquareProps & import("react").RefAttributes<HTMLHRElement>>;
