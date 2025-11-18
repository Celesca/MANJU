import { type ComponentProps, type ReactEventHandler, type ReactNode } from "react";
import type { ThemingProps } from "../../types";
export interface PaginationButtonTheme {
    base: string;
    active: string;
    disabled: string;
}
export interface PaginationButtonProps extends ComponentProps<"button">, ThemingProps<PaginationButtonTheme> {
    active?: boolean;
    children?: ReactNode;
    className?: string;
    onClick?: ReactEventHandler<HTMLButtonElement>;
}
export interface PaginationPrevButtonProps extends Omit<PaginationButtonProps, "active"> {
    disabled?: boolean;
}
export declare const PaginationButton: import("react").ForwardRefExoticComponent<Omit<PaginationButtonProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
export declare function PaginationNavigation({ children, className, onClick, disabled, theme: customTheme, clearTheme, applyTheme, ...props }: PaginationPrevButtonProps): import("react/jsx-runtime").JSX.Element;
export declare namespace PaginationNavigation {
    var displayName: string;
}
