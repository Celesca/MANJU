import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface ModalFooterTheme {
    base: string;
    popup: string;
}
export interface ModalFooterProps extends ComponentProps<"div">, ThemingProps<ModalFooterTheme> {
}
export declare const ModalFooter: import("react").ForwardRefExoticComponent<Omit<ModalFooterProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
