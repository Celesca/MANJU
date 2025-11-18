import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, ThemingProps } from "../../types";
export interface CheckboxTheme {
    base: string;
    color: FlowbiteColors;
    indeterminate: string;
}
export interface CheckboxProps extends Omit<ComponentProps<"input">, "type" | "ref" | "color">, ThemingProps<CheckboxTheme> {
    color?: DynamicStringEnumKeysOf<FlowbiteColors>;
    indeterminate?: boolean;
}
export declare const Checkbox: import("react").ForwardRefExoticComponent<CheckboxProps & import("react").RefAttributes<HTMLInputElement>>;
