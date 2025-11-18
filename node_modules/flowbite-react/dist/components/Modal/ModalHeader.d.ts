import { type ComponentProps, type ElementType } from "react";
import type { ThemingProps } from "../../types";
export interface ModalHeaderTheme {
    base: string;
    popup: string;
    title: string;
    close: {
        base: string;
        icon: string;
    };
}
export interface ModalHeaderProps extends ComponentProps<"div">, ThemingProps<ModalHeaderTheme> {
    as?: ElementType;
}
export declare const ModalHeader: import("react").ForwardRefExoticComponent<Omit<ModalHeaderProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
