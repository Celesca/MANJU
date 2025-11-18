import { type ComponentProps, type ReactNode } from "react";
import type { ThemingProps } from "../../types";
import type { PaginationButtonProps, PaginationButtonTheme } from "./PaginationButton";
export interface PaginationTheme {
    base: string;
    layout: PaginationLayoutTheme;
    pages: PaginationPagesTheme;
}
export interface PaginationRootTheme {
    base: string;
}
export interface PaginationLayoutTheme {
    table: {
        base: string;
        span: string;
    };
}
export interface PaginationPagesTheme {
    base: string;
    showIcon: string;
    previous: PaginationNavigationTheme;
    next: PaginationNavigationTheme;
    selector: PaginationButtonTheme;
}
export interface PaginationNavigationTheme {
    base: string;
    icon: string;
}
export interface BasePaginationProps extends ComponentProps<"nav">, ThemingProps<PaginationTheme> {
    layout?: "navigation" | "pagination" | "table";
    currentPage: number;
    nextLabel?: string;
    onPageChange: (page: number) => void;
    previousLabel?: string;
    showIcons?: boolean;
}
export interface DefaultPaginationProps extends BasePaginationProps {
    layout?: "navigation" | "pagination";
    renderPaginationButton?: (props: PaginationButtonProps) => ReactNode;
    totalPages: number;
}
export interface TablePaginationProps extends BasePaginationProps {
    layout: "table";
    itemsPerPage: number;
    totalItems: number;
}
export type PaginationProps = DefaultPaginationProps | TablePaginationProps;
export declare const Pagination: import("react").ForwardRefExoticComponent<(Omit<DefaultPaginationProps, "ref"> | Omit<TablePaginationProps, "ref">) & import("react").RefAttributes<HTMLElement>>;
