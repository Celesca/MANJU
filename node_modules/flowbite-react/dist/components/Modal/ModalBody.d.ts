import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface ModalBodyTheme {
    base: string;
    popup: string;
}
export interface ModalBodyProps extends ComponentProps<"div">, ThemingProps<ModalBodyTheme> {
}
export declare const ModalBody: import("react").ForwardRefExoticComponent<Omit<ModalBodyProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
