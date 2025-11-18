import { type ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteStateColors, ThemingProps } from "../../types";
export interface LabelTheme {
    root: LabelRootTheme;
}
export interface LabelRootTheme {
    base: string;
    colors: LabelColors;
    disabled: string;
}
export interface LabelColors extends FlowbiteStateColors {
    [key: string]: string;
    default: string;
}
export interface LabelProps extends Omit<ComponentProps<"label">, "color">, ThemingProps<LabelTheme> {
    color?: DynamicStringEnumKeysOf<LabelColors>;
    disabled?: boolean;
}
export declare const Label: import("react").ForwardRefExoticComponent<Omit<LabelProps, "ref"> & import("react").RefAttributes<HTMLLabelElement>>;
