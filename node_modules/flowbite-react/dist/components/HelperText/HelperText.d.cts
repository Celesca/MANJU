import { type ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, ThemingProps } from "../../types";
export interface HelperTextTheme {
    root: HelperTextRootTheme;
}
export interface HelperTextRootTheme {
    base: string;
    colors: HelperColors;
}
export interface HelperColors extends Pick<FlowbiteColors, "gray" | "info" | "failure" | "warning" | "success"> {
    [key: string]: string;
}
export interface HelperTextProps extends Omit<ComponentProps<"p">, "color">, ThemingProps<HelperTextTheme> {
    color?: DynamicStringEnumKeysOf<HelperColors>;
}
export declare const HelperText: import("react").ForwardRefExoticComponent<Omit<HelperTextProps, "ref"> & import("react").RefAttributes<HTMLParagraphElement>>;
