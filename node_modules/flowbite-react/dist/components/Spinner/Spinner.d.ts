import { type ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, FlowbiteSizes, ThemingProps } from "../../types";
export interface SpinnerTheme {
    base: string;
    color: SpinnerColors;
    light: {
        off: {
            base: string;
            color: SpinnerColors;
        };
        on: {
            base: string;
            color: SpinnerColors;
        };
    };
    size: SpinnerSizes;
}
export interface SpinnerColors extends Pick<FlowbiteColors, "failure" | "gray" | "info" | "pink" | "purple" | "success" | "warning"> {
    [key: string]: string;
    default: string;
}
export interface SpinnerSizes extends Pick<FlowbiteSizes, "xs" | "sm" | "md" | "lg" | "xl"> {
    [key: string]: string;
}
export interface SpinnerProps extends Omit<ComponentProps<"span">, "color">, ThemingProps<SpinnerTheme> {
    color?: DynamicStringEnumKeysOf<SpinnerColors>;
    light?: boolean;
    size?: DynamicStringEnumKeysOf<SpinnerSizes>;
}
export declare const Spinner: import("react").ForwardRefExoticComponent<Omit<SpinnerProps, "ref"> & import("react").RefAttributes<HTMLSpanElement>>;
