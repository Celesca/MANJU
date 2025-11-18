import type { ComponentPropsWithoutRef } from "react";
import type { ThemingProps } from "../../types";
import { type FloatingLabelTheme } from "./theme";
export type FloatingLabelColor = "default" | "success" | "error";
export type FloatingLabelSizing = "sm" | "md";
export type FloatingLabelVariant = "filled" | "outlined" | "standard";
export interface FloatingLabelProps extends ComponentPropsWithoutRef<"input">, ThemingProps<FloatingLabelTheme> {
    label: string;
    color?: FloatingLabelColor;
    sizing?: FloatingLabelSizing;
    variant: FloatingLabelVariant;
    disabled?: boolean;
}
export declare const FloatingLabel: import("react").ForwardRefExoticComponent<FloatingLabelProps & import("react").RefAttributes<HTMLInputElement>>;
