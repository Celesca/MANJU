import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface ToastToggleTheme {
    base: string;
    icon: string;
}
export interface ToastToggleProps extends ComponentProps<"button">, ThemingProps<ToastToggleTheme> {
    xIcon?: FC<ComponentProps<"svg">>;
    onDismiss?: () => void;
}
export declare const ToastToggle: import("react").ForwardRefExoticComponent<Omit<ToastToggleProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
