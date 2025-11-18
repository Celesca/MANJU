import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { ButtonProps } from "../Button/Button";
export interface ButtonGroupTheme {
    base: string;
}
export interface ButtonGroupProps extends ComponentProps<"div">, Pick<ButtonProps, "outline" | "pill">, ThemingProps<ButtonGroupTheme> {
}
export declare const ButtonGroup: import("react").ForwardRefExoticComponent<Omit<ButtonGroupProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
