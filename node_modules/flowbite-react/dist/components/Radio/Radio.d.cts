import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, ThemingProps } from "../../types";
export interface RadioTheme {
    base: string;
    color: FlowbiteColors;
}
export interface RadioProps extends Omit<ComponentProps<"input">, "ref" | "type">, ThemingProps<RadioTheme> {
    color?: DynamicStringEnumKeysOf<FlowbiteColors>;
}
export declare const Radio: import("react").ForwardRefExoticComponent<RadioProps & import("react").RefAttributes<HTMLInputElement>>;
