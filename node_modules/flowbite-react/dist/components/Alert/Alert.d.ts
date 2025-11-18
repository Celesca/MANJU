import { type ComponentProps, type FC, type ReactNode } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, ThemingProps } from "../../types";
export interface AlertTheme {
    base: string;
    borderAccent: string;
    closeButton: AlertCloseButtonTheme;
    color: FlowbiteColors;
    icon: string;
    rounded: string;
    wrapper: string;
}
export interface AlertCloseButtonTheme {
    base: string;
    color: FlowbiteColors;
    icon: string;
}
export interface AlertProps extends Omit<ComponentProps<"div">, "color">, ThemingProps<AlertTheme> {
    additionalContent?: ReactNode;
    color?: DynamicStringEnumKeysOf<FlowbiteColors>;
    icon?: FC<ComponentProps<"svg">>;
    onDismiss?: ComponentProps<"button">["onClick"];
    rounded?: boolean;
    withBorderAccent?: boolean;
}
export declare const Alert: import("react").ForwardRefExoticComponent<Omit<AlertProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
